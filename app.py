import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Wine Quality Checker :wine_glass:')
st.markdown('A RF model to classify wine based on its quality.\n')

st.header("Wine Features")

fixed_acidity = st.number_input("Enter fixed acidity",format="%.2f")

volatile_acidity = st.number_input("Enter volatile acidity",format="%.2f")

citric_acid = st.number_input("Enter citric acid",format="%.2f")

residual_sugar = st.number_input("Enter residual sugar",format="%.2f")

chlorides = st.number_input("Enter chlorides",format="%.2f")

free_sulfur_dioxide = st.number_input("Enter free sulfur dioxide",format="%.2f")

total_sulfur_dioxide = st.number_input("Enter total sulfur dioxide",format="%.2f")

density = st.number_input("Enter density",format="%.2f")

pH = st.number_input("Enter pH",format="%.2f")

sulphates = st.number_input("Enter sulphates",format="%.2f")

alcohol = st.number_input("Enter alcohol", format="%.2f")

input_data = (fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

st.text('')
if st.button("Check"):
    model = joblib.load("rf_model.sav")
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 1: 
        st.success('Good Quality :thumbsup:')
    else: 
        st.error('Bad Quality :thumbsdown:') 
    
