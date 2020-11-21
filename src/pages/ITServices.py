"""IT Service Management  Models will be displayed here """
import streamlit as st
import shared.components

# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        shared.components.title_awesome("")
        st.write(
            """
            The following IT Service Management models will be implemented

            1. Ticket Sentiments 

            """
        )