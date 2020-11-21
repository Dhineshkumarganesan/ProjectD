"""Financial Models will be displayed here """
import streamlit as st
import shared.components

# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        shared.components.title_awesome("")
        st.write(
            """
            The following financial models will be implemented

            1.Currency Exchange Prediction
            2.Loan Approval Prediction
            3.Detect Fradulent Cases
            4.Forecast Stock Prices
            5.Bank Market Segmentation
            6.Invoice Classification
            7.Invoice Segmentation

    """
    )