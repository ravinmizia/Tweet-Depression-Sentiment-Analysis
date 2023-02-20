import streamlit as st
import eda 
import prediction


navigation = st.sidebar.selectbox('Page : ', ('Explore Data', 'Predict Tweet Sentiment'))


if navigation == 'Explore Data':
    eda.run()
else:
    prediction.run()
    
