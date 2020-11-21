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



# load Vectorizer For Gender Prediction
news_vectorizer = open("src/pages/Services/models/final_news_cv_vectorizer.pkl","rb")
news_cv = joblib.load(news_vectorizer)



# # load Model For Gender Prediction
# news_nv_model = open("models/naivebayesgendermodel.pkl","rb")
# news_clf = joblib.load(news_nv_model)





def write():
	"""News Classifier"""
	st.title("Forecast Stock Prices")
	# st.subheader("ML App with Streamlit")
	





	st.sidebar.subheader("About")




#if __name__ == '__main__':
#	main()

