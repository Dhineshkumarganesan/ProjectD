import streamlit as st

import numpy as np
import pandas as pd
from sklearn.linear_model import  LinearRegression
from  sklearn.model_selection import  train_test_split

import time
import warnings
import shared.components
import os


def write():

    outname = 'data.csv'

    outdir = 'src/pages/Services/data/'
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)    
    data = pd.read_csv(fullname)

    st.title('Predict The House Price')

    show_data = st.checkbox('See the House Data')

    if show_data:
        st.write(data.head())


    sqft = st.slider("Square Feet of house Required :",int(data.sqft_living.min()),int(data.sqft_living.max()),int(data.sqft_living.mean()))

    bath = st.slider("No. of Bathrooms :",int(data.bathrooms.min()),int(data.bathrooms.max()),int(data.bathrooms.mean()))
    bed = st.slider("No. of Bedrooms :",int(data.bedrooms.min()),int(data.bedrooms.max()),int(data.bedrooms.mean()))
    floor = st.slider("No. of Floors:",int(data.floors.min()),int(data.floors.max()),int(data.floors.mean()))


    X = data.drop('price',axis=1)
    y = data.price

    model = LinearRegression()

    model.fit(X,y)

    pred = model.predict([[sqft,bath,bed,floor]])[0]





    if st.button("Get House Price"):
        st.header("Estimated House Price is USD {}".format(int(pred)))