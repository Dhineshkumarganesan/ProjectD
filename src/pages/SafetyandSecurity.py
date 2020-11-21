"""Safety and Security Models will be displayed here """
import streamlit as st
import shared.components

# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        shared.components.title_awesome("")
        st.write(
            """
            The following Safety and Security models will be implemented

            1.Face Mask Detection
            2.Fire Hazard Detection
            3.Drowning Detection
          
    """
    )