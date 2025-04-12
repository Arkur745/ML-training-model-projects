from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open(model_path,'rb') as file:
    model = pickle.load(file)

with open('encoder.pkl','rb') as file:
    encoder = pickle.load(file)


genders = ['male','female']

st.title('Calories Burnt Predictor')

gender=st.selectbox('Gender',genders)
age=st.slider('Age',25,79,30)
height=st.slider('Height(in cm)',140,222,160)
weight=st.slider('weight',36,132,65)
duration=st.slider('Duration',1,30,10)
heart=st.slider('Heart_Rate',67,128,100)
temp=st.slider("Body_temp",37,41,38)

input_data = pd.DataFrame({
    'Gender':[gender],
    'Age' :[age],
    'Height':[height],
    'Weight':[weight],
    'Duration':[duration],
    'Heart_Rate':[heart],
    'Body_Temp':[temp]
})
gender_encoded = encoder.transform(input_data[['Gender']])
gender_encoded_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))
input_data_encoded = pd.concat([input_data.drop('Gender', axis=1), gender_encoded_df], axis=1)
prediction = model.predict(input_data_encoded)

st.write(f'The calories burnt are:{prediction} ')

