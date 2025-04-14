import numpy as np
import pandas as pd
import streamlit as st
import tensorflow
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model('project4/model.h5')
with open('project4/tokenizer.pkl','rb') as handle:
    tokenizer=pickle.load(handle)

st.title('Mental Health Classifier')
input_text=st.text_input('Enter the text')
if st.button('Assess'):
    seq = tokenizer.texts_to_sequences([input_text])
    pad = pad_sequences(seq,maxlen=200,padding='post',truncating='post')

    pred = model.predict(pad)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    # Class mapping
    class_name = {
        0: "Stress",
        1: "Depression",
        2: "Bipolar Disorder",
        3: "Personality Disorder",
        4: "Anxiety"
    }

    # Threshold check
    if confidence < 0.5:
        st.write("Your thoughts don't clearly match any specific mental health condition.")
        st.write("You might be doing okay, or it's too ambiguous to tell from the input.")
    else:
        st.write(f"Predicted Condition: **{class_name[predicted_class]}**")
        st.write(f"Confidence: {confidence:.2f}")

