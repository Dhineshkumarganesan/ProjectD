"""Main module for the streamlit app"""
import streamlit as st
import src.pages.about
import src.pages.home
import src.pages.Finance
import src.pages.Automobile
import src.pages.ITServices
import src.pages.SafetyandSecurity
import src.pages.Services.ticketsentiments
import src.pages.Services.financialanalyser
import src.pages.Services.loanapprovalprediction
import src.pages.Services.currencyexchangeprediction
import src.pages.Services.detectFradulentCases
import src.pages.Services.forecaststockprices
import src.pages.Services.bankmarketsegmentation
import src.pages.Services.housepriceprediction
import src.pages.Services.invoiceclassification
import src.pages.Services.invoicesegmentation
import src.pages.Services.sentimentswithemoji
import src.pages.Services.sladashboard
import src.pages.Services.tickettrends
import src.pages.Services.facedetection
import src.pages.Services.vehicleprediction
import src.pages.Services.pedestriandetection
import src.pages.Services.damagedetection
import src.pages.Services.imageclassifier
import src.pages.Services.facemaskdetection
import src.pages.Services.firedetection
import src.pages.Services.drowningdetection


import shared.components

PAGES = {
    "Home": src.pages.home,
    "Finance": src.pages.Finance,
    "Automobile": src.pages.Automobile,
    "Services": src.pages.ITServices,
    "SafetyandSecurity": src.pages.SafetyandSecurity,
    "About": src.pages.about,
    
}

FINANCEPAGES = {
    "Sales Prediction" : src.pages.Services.financialanalyser,
    "Loan Approval Prediction" : src.pages.Services.loanapprovalprediction,
    "Currency Exchange Prediction" : src.pages.Services.currencyexchangeprediction,
    "Detect Fradulent Cases" : src.pages.Services.detectFradulentCases,
    "Forecast Stock Prices" : src.pages.Services.forecaststockprices,
    "Bank Market Segmentation" : src.pages.Services.bankmarketsegmentation,
    "House Price Prediction" :  src.pages.Services.housepriceprediction,
    "Invoice Classification" :  src.pages.Services.invoiceclassification,
    "Invoice Segmentation" :  src.pages.Services.invoicesegmentation  
}

SERVICESPAGES = {
    "Ticket Sentiments" : src.pages.Services.ticketsentiments,
    "Sentiments with Emoji" : src.pages.Services.sentimentswithemoji,
    "SLA Dashboard" : src.pages.Services.sladashboard,
    "Ticket Trends" : src.pages.Services.tickettrends,
    
}

AUTOMOBILEPAGES = {
    "Face Detection" : src.pages.Services.facedetection,
    "Vehicle Detection" : src.pages.Services.vehicleprediction,
    "Pedestrian Detection" : src.pages.Services.pedestriandetection,
    "Damage Detection" : src.pages.Services.damagedetection,
    "Image Classifier" : src.pages.Services.imageclassifier,
    
}

SAFETYSECURITYPAGES = {
    "Face Mask Detection" : src.pages.Services.facemaskdetection,
    "Fire Detection" : src.pages.Services.firedetection,
    "Drowning Detection" : src.pages.Services.drowningdetection,
    
}

def main():
    """Main function"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]


    if selection == 'Home':
        st.sidebar.title("About")
        st.sidebar.info(
            "Project D  "
            "AIML Use Cases "
            #"[issues](https://github.com/MarcSkovMadsen/awesome-streamlit/issues) of or "
            #"[pull requests](https://github.com/MarcSkovMadsen/awesome-streamlit/pulls) "
            #"to the [source code](https://github.com/MarcSkovMadsen/awesome-streamlit). "
        )
        #st.sidebar.title("About")
        #st.sidebar.info(
        #    """
        #        Machine Learning & Artificial Implementation using Streamlit
            
        #    """
        #)
        with st.spinner(f"Loading {selection} ..."):
            shared.components.write_page(page)           
    elif selection == "Finance":
        st.sidebar.title("Financial Models")   
        Finance_modes = st.sidebar.selectbox("Please select a Model", ["Sales Prediction",
                                                             "Loan Approval Prediction",
                                                             "Currency Exchange Prediction",
                                                             "Detect Fradulent Cases",
                                                             "Forecast Stock Prices",
                                                             "House Price Prediction",
                                                             "Invoice Classification",
                                                             "Invoice Segmentation",
                                                             "Bank Market Segmentation"])
        page = FINANCEPAGES[Finance_modes]
        with st.spinner(f"Loading {selection} ..."):
            shared.components.write_page(page)                                                           
    elif selection == "Automobile":
        st.sidebar.title("Computer Vision Models")
        Automobile_modes = st.sidebar.selectbox("Please select a Model", ["Face Detection",
                                                             "Damage Detection",
                                                             "Image Classifier",
                                                             "Pedestrian Detection",
                                                             "Vehicle Detection"])
        page = AUTOMOBILEPAGES[Automobile_modes]
        with st.spinner(f"Loading {selection} ..."):
            shared.components.write_page(page)        
    elif selection == "Services":
        st.sidebar.title("Ticket Analysis")
        ticket_mode = st.sidebar.selectbox("Please select a Model", ["Ticket Sentiments",
                                                             "Sentiments with Emoji",
                                                             "SLA Dashboard",
                                                             "Ticket Trends"])
        page = SERVICESPAGES[ticket_mode]
        with st.spinner(f"Loading {ticket_mode} ..."):
            shared.components.write_page(page)                                                             
    elif selection == "SafetyandSecurity":
        st.sidebar.title("CV on Safety & Security ")
        SafetyandSecurity_mode = st.sidebar.selectbox("Please select a Model", ["Face Mask Detection",
                                                             "Fire Detection",
                                                             "Drowning Detection"])
        page = SAFETYSECURITYPAGES[SafetyandSecurity_mode]
        with st.spinner(f"Loading {SafetyandSecurity_mode} ..."):
            shared.components.write_page(page)     

    
        
 


        


if __name__ == "__main__":
    main()
