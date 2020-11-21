import streamlit as st 
import warnings
import pandas as pd
import warnings
import shared.components
import os
import tempfile
import cv2
#warnings.filterwarnings("ignore")



def write():
	st.title('PEDESTRIAN DETECTION ')
	fped = st.file_uploader("Upload Pedestrian video")
	if fped:
		tfileped=tempfile.NamedTemporaryFile(delete=False)	
		tfileped.write(fped.read())
		
		vfped = cv2.VideoCapture(tfileped.name)

		stframeped = st.empty()
		ped_cascade = cv2.CascadeClassifier('src/pages/Services/frecog/haarcascade_fullbody.xml')
		while vfped.isOpened():
			ret, frameped = vfped.read()
			if not ret:
					print("Can't receive frame (stream end?). Exiting ...")
					break
			gray = cv2.cvtColor(frameped, cv2.COLOR_BGR2GRAY)
			cars = ped_cascade.detectMultiScale(gray,1.1,1)
			color = (0, 255, 0)	
			for (x, y, w, h) in cars:
				cv2.rectangle(frameped,(x,y),(x+w,y+h),(255,0,0),2)
				cv2.putText(frameped,"Human",(x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			if cv2.waitKey(33)==13: 
				break
			stframeped.image(frameped)

            
          




#if __name__ == '__main__':
#	main()

