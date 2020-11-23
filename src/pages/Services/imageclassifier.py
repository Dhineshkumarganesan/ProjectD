import streamlit as st 
from PIL import Image
#from classify import predict
import shared.components
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
#from tensorflow.python.keras.applications.vgg16 import preprocess_input
#from tensorflow.python.keras.applications.vgg16 import decode_predictions
#from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.applications.mobilenet import decode_predictions
from tensorflow.python.keras.applications.mobilenet import MobileNet
#from tensorflow.python.keras.applications.xception  import preprocess_input
#from tensorflow.python.keras.applications.xception  import decode_predictions
#from tensorflow.python.keras.applications.xception  import Xception

def write():
    st.title("Image Classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = predict(uploaded_file)
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))


def predict(image1): 
    model = MobileNet()
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label 