"""About me """
import streamlit as st
import shared.components

# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        shared.components.title_awesome("")
        st.write(
            """
            I am Dhinesh Kumar Ganeshan with PG in AIML, AWS ML Certified with 15 years of Experience 

            """
        )