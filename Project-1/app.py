
import streamlit as st
import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("Fake News Detection System")

news = st.text_area("Enter news text:")

if st.button("Predict"):
    
    news_vector = vectorizer.transform([news])
    
    prediction = model.predict(news_vector)
    
    if prediction[0] == 1:
        st.success("This is Real News")
    else:
        st.error("This is Fake News")