import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman', {'Predictor', 'EDA'})

if navigation == 'Predictor':
    prediction.run()
else:
    eda.run()