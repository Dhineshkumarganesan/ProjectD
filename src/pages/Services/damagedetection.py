import streamlit as st 
import joblib,os
import spacy
import warnings
import pandas as pd
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
import shared.components

#warnings.filterwarnings("ignore")






def write():
	"""News Classifier"""
	st.title("Damage Detection")
	# st.subheader("ML App with Streamlit")
	



	st.sidebar.subheader("About")




#if __name__ == '__main__':
#	main()

