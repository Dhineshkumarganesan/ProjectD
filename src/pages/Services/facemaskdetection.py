import streamlit as st
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import numpy as np
import pandas as pd
import shared.components
import cv2
import os
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from PIL import Image
from enum import Enum
from io import BytesIO, StringIO
from typing import Union

FILE_TYPES = ["csv", "py", "png", "jpg"]


class FileType(Enum):
    """Used to distinguish between file types"""

    IMAGE = "Image"
    CSV = "csv"
    PYTHON = "Python"



def get_file_name(file: Union[BytesIO, StringIO], folder_path='C:\\ProjD\\Assets\\'):
    if isinstance(file, BytesIO):
        return os.path.join(folder_path,file.name)

def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
    """The file uploader widget does not provide information on the type of file uploaded so we have
    to guess using rules or ML. See
    [Issue 896](https://github.com/streamlit/streamlit/issues/896)

    I've implemented rules for now :-)

    Arguments:
        file {Union[BytesIO, StringIO]} -- The file uploaded

    Returns:
        FileType -- A best guess of the file type
    """

    if isinstance(file, BytesIO):
        return FileType.IMAGE
    content = file.getvalue()
    if (
        content.startswith('"""')
        or "import" in content
        or "from " in content
        or "def " in content
        or "class " in content
        or "print(" in content
    ):
        return FileType.PYTHON

    return FileType.CSV

@st.cache(hash_funcs={cv2.dnn_Net: hash})
def load_face_detector_and_model():
    prototxt_path = os.path.sep.join(["C:\\ProjD\\25FinD\\src\\pages\\Services\\face_detector", "deploy.prototxt"])
    weights_path = os.path.sep.join(["C:\\ProjD\\25FinD\\src\\pages\\Services\\face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    cnn_net = cv2.dnn.readNet(prototxt_path, weights_path)

    return cnn_net

@st.cache(allow_output_mutation=True)
def load_cnn_model():
    cnn_model = load_model("C:\\ProjD\\25FinD\\src\\pages\\Services\\models\\mask_detector.model")

    return cnn_model

def write():
    st.title(' Face Mask Detector')

    net = load_face_detector_and_model()
    model = load_cnn_model()

    selected_option = st.radio("Choose", ('File','Webcam'))

    if selected_option == 'File':
        #uploaded_image = st.sidebar.file_uploader("Choose a JPG file", type="jpg")
        uploaded_image = st.sidebar.file_uploader("Choose a JPG file", type=FILE_TYPES)
        confidence_value = st.sidebar.slider('Confidence:', 0.0, 1.0, 0.5, 0.1)
        if uploaded_image:
            image1 = Image.open(uploaded_image)
            st.sidebar.image(image1, caption='Uploaded Image.', use_column_width=True)
            #st.sidebar.info('Uploaded image:')
            #st.sidebar.image(uploaded_image, width=240)
            #f = open(uploaded_image, 'rb')
        
                #file = st.file_uploader("Upload file", type=FILE_TYPES)
            show_file = st.empty()
            if not uploaded_image:
                show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
                return
            file_type = get_file_type(uploaded_image)
            if file_type == FileType.IMAGE:
                show_file.image(image1)
            elif file_type == FileType.PYTHON:
                st.code(uploaded_image.getvalue())
            else:
                data = pd.read_csv(uploaded_image)
                st.dataframe(data.head(10))
            
            f = open(get_file_name(uploaded_image), 'rb')
            img_bytes = f.read()
            f.close()
            

            grad_cam_button = st.sidebar.button('Grad CAM')
            patch_size_value = st.sidebar.slider('Patch size:', 10, 90, 20, 10)
            occlusion_sensitivity_button = st.sidebar.button('Occlusion Sensitivity')
            image = cv2.imdecode(np.fromstring(img_bytes, np.uint8), 1)
            #image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig = image.copy()
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                        (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_value:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    face = image[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    expanded_face = np.expand_dims(face, axis=0)

                    (mask, withoutMask) = model.predict(expanded_face)[0]

                    predicted_class = 0
                    label = "No Mask"
                    if mask > withoutMask:
                        label = "Mask"
                        predicted_class = 1

                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    cv2.putText(image, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                    st.image(image, width=640)
                    st.write('### ' + label)

            if grad_cam_button:
                data = ([face], None)
                explainer = GradCAM()
                grad_cam_grid = explainer.explain(
                    data, model, class_index=predicted_class, layer_name="Conv_1"
                )
                st.image(grad_cam_grid)

            if occlusion_sensitivity_button:
                data = ([face], None)
                explainer = OcclusionSensitivity()
                sensitivity_occlusion_grid = explainer.explain(data, model, predicted_class, patch_size_value)
                st.image(sensitivity_occlusion_grid)
   
    # PROGRAM FOR WEB CAM
    if selected_option == 'Webcam':
        labels_dict={0:'without_mask',1:'with_mask'}
        color_dict={0:(0,0,255),1:(0,255,0)}
        size = 4
        
        webcam = cv2.VideoCapture(0) #Use camera 0
        st.write("Webcam On")
        stframe_cam = st.empty()
        # We load the xml file
        classifier = cv2.CascadeClassifier('src/pages/Services/frecog/haarcascade_frontalface_default.xml')

        while True:
            (rval, im) = webcam.read()
            stframe_cam.image(im)
           # st.write("Webcam Read")
            #if im:
            #ret, framecar = vf.read()
            im = cv2.flip(im,1,1) #Flip to act as a mirror
                
            # Resize the image to speed up detection
            mini = cv2.resize(im,(im.shape[1]//size, im.shape[0]//size))

            # detect MultiScale / faces 
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:
                (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
                #Save just the rectangle faces in SubRecFaces
                face_img = im[y:y+h, x:x+w]
                resized=cv2.resize(face_img,(150,150))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,150,150,3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)
                #print(result)
                    
                label=np.argmax(result,axis=1)[0]
                
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                    
                # Show the image
                stframe_cam.image('LIVE',   im)
                #cv2.imshow('LIVE',   im)
                key = cv2.waitKey(10)
                # if Esc key is press then break out of the loop 
                if key == 27: #The Esc key
                    break
        # Stop video
        webcam.release()

        # Close all started windows
        cv2.destroyAllWindows()
#write()
#uploaded_image.close()