import streamlit as st 
import warnings
import pandas as pd
import warnings
import shared.components
import os
import tempfile
import cv2
import time
#warnings.filterwarnings("ignore")



def write():
	st.title('VEHICLE DETECTION ')
	fcar = st.file_uploader("Upload Vehicle video")
	if fcar:
		tfile=tempfile.NamedTemporaryFile(delete=False)	
		tfile.write(fcar.read())
		
		vf = cv2.VideoCapture(tfile.name)
		vf.set(cv2.CAP_PROP_FPS, 25)

		stframe = st.empty()
		stframe_1 = st.empty()
		cars_cascade = cv2.CascadeClassifier('src/pages/Services/frecog/cars.xml')
		while vf.isOpened():
			ret, framecar = vf.read()
			if not ret:
					print("Can't receive frame (stream end?). Exiting ...")
					break
			gray = cv2.cvtColor(framecar, cv2.COLOR_BGR2GRAY)
			cars = cars_cascade.detectMultiScale(gray,1.1,1)
				
			for (x, y, w, h) in cars:
				cv2.rectangle(framecar,(x,y),(x+w,y+h),(255,0,0),2)
			if cv2.waitKey(33)==13: 
				break
			stframe_1.image(framecar, channels="BGR")
			time.sleep(0.01)
			video_byte = open(tfile.name, 'rb').read()
			#video_byte = framecar
			stframe.video(video_byte)

# /*
# 		# Try Webcam diplay
# 		labels_dict={0:'without_mask',1:'with_mask'}
# 		color_dict={0:(0,0,255),1:(0,255,0)}

# 		size = 4
# 		vfcam = cv2.VideoCapture(0)

# 		stframecam = st.empty()
# 		classifier = cv2.CascadeClassifier('C:\\ProjD\\25FinD\\src\\pages\\Services\\frecog\\haarcascade_frontalface_default.xml')
        
          
# 		while True:
# 			(rval, im) = vfcam.read()
# 			im=cv2.flip(im,1,1) #Flip to act as a mirror

# 			# Resize the image to speed up detection
# 			mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

# 			# detect MultiScale / faces 
# 			faces = classifier.detectMultiScale(mini)

# 			# Draw rectangles around each face
# 			for f in faces:
# 				(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
# 				#Save just the rectangle faces in SubRecFaces
# 				face_img = im[y:y+h, x:x+w]
# 				resized=cv2.resize(face_img,(150,150))
# 				normalized=resized/255.0
# 				reshaped=np.reshape(normalized,(1,150,150,3))
# 				reshaped = np.vstack([reshaped])
# 				result=model.predict(reshaped)
# 				#print(result)
				
# 				label=np.argmax(result,axis=1)[0]
			
# 				cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
# 				cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
# 				cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
				
# 			# Show the image
# 			stframe.image(cv2.imshow('LIVE',   im))
# 			key = cv2.waitKey(10)
# 			# if Esc key is press then break out of the loop 
# 			if key == 27: #The Esc key
# 				break
		
# 		vfcam.release()		
# */
#if __name__ == '__main__':
#	main()

