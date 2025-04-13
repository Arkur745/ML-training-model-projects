import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

with open('model.pkl','rb') as file:
    model = pickle.load(file)
st.title("Titanic Survival Prediction")


classes =[1,2,3]
genders=['male','female']
port_dict = {
    'Cherbourg': 'C',
    'Queenstown': 'Q',
    'Southampton': 'S'
}


pclass = st.selectbox('Select Class',classes)
sex = st.selectbox('Gender',genders)
age = st.slider('Age',1,80,18)
sib = st.slider('Siblings and Spouse Aboard',0,8,2)
parch = st.slider('Parents and children Aboard',0,6,2)
fare = st.number_input('Enter the Fare',0,512)
selected = st.selectbox(
    'Select the port of embarkation:',
    list(port_dict.keys())  
)
emb = port_dict[selected]


df = pd.DataFrame({
    'Pclass' : [pclass],
    'Sex' : [sex],
    'Age' : [age],
    'SibSp':[sib],
    'Parch':[parch],
    'Fare':[fare],
    'Embarked':[emb]
})

df['Sex'] = np.where(df['Sex'] == "male",1,0)
pd.set_option('future.no_silent_downcasting', True)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

scaler = StandardScaler()
df[['Age','Fare','Pclass']]=scaler.fit_transform(df[['Age','Fare','Pclass']])



predicted = model.predict(df)

if predicted == 0:
    st.write(f'The passenger didn\'t survived')
elif predicted==1:
    st.write(f'The passenger survived')