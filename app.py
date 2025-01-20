import streamlit as st
import pickle 

cv = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Classification ")
st.write("This is a Machine Learning application to classify/detect  SMS as Spam or Not Spam ")
st.write("Made by Ananya Rai ")

input_sms = st.text_area("Enter the SMS",height=100)

if st.button("Classify"):
    if input_sms:
        data =[input_sms]
        vector_input = cv.transform(data).toarray()
        result = model.predict(vector_input)
        if result[0] == 0:
            st.write("Spam")
        else:
            st.write("Not Spam")
    else:
        st.write("Please Enter Message to Predict")

