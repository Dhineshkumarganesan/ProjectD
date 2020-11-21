import streamlit as st 
import warnings
import pandas as pd
import warnings
import shared.components
import os
import tempfile
import cv2
import numpy as np
import joblib
import pytesseract
from PIL import Image
#warnings.filterwarnings("ignore")

# load invoice classification model
path ="src/pages/Services/models/"

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def write():
	st.title('INVOICE SEGMENTATION - WORK IN PROGRESS ')
	fcar = st.file_uploader("Upload Invoice in JPG or JPEG Format ")
	if fcar:
		tfile=tempfile.NamedTemporaryFile(delete=False)	
		tfile.write(fcar.read())

		# Loading a Image
		load_img = Image.open(fcar)
		st.image(load_img, caption='Uploaded Image.', use_column_width=True)
		text = pytesseract.image_to_string(load_img)
		text1 = [text]


		Inv_button = st.button('Predict')
		# Predict on given text
		if Inv_button:
			predictor = load_prediction_models(path +"invoiceclassification_model.pkl")
			pred = predictor.predict(text1)
			st.label("It is RECEIPT" if pred == 1 else "Not a RECEIPT")

#if __name__ == '__main__':
#	main()

