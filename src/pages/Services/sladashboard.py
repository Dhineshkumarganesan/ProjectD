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

path ="src/pages/Services/models/"

# load Vectorizer For Gender Prediction
news_vectorizer = open(path +"final_news_cv_vectorizer.pkl","rb")
news_cv = joblib.load(news_vectorizer)



# # load Model For Gender Prediction
# news_nv_model = open("models/naivebayesgendermodel.pkl","rb")
# news_clf = joblib.load(news_nv_model)

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model



# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



def write():
	"""News Classifier"""
	st.title("SLA Dashboard")
	# st.subheader("ML App with Streamlit")
	







	st.sidebar.subheader("About")




#if __name__ == '__main__':
#	main()

