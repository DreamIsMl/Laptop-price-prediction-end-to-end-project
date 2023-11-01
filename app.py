import pickle

import numpy as np
import streamlit as st

with open('F:\\class\\Machine Learning\\project\\Car price prediction\\pipe.pkl','rb')as model_file:
    pipe = pickle.load(model_file)
    
with open('F:\class\Machine Learning\project\Car price prediction\df.pkl','rb') as df_file:
    df = pickle.load(df_file)

name = st.selectbox('Car Name',df['name'].unique())
company = st.selectbox('Company Name',df['company'].unique())
year= st.selectbox('Car Year',df['year'].unique())
kms_driven = st.number_input("Enter kms as meter")
fuel = st.selectbox('Fuel Type',df['fuel_type'].unique())

if st.button('prediction Price'):
    query = np.array([name,company,year,kms_driven,fuel])
    query = query.reshape(1,5)
    st.title('The Prediction Price of This Car is ' + str(int(np.exp(pipe.predict(query)[0]))))