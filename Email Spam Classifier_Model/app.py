import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


td = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))


def transform_text(text):
    text = text.lower()  # Lower Case
    text = nltk.word_tokenize(text)  # Tokenization
    y = []  # Assigning Y list

    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]  # Transferring y value into text variable
    y.clear()  # clearing Y

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:  # Removing Stop Words and Punctuation
            y.append(i)
    text = y[:]  # Transferring y value into text variable
    y.clear()  # Clearing Y

    for i in text:
        y.append(ps.stem(i))  # Stemming

    return " ".join(y)


st.title("Email Spam Classifier")

input_sms=st.text_area("Enter your message")

if st.button("Predict"):

    #1.preprocess
    transformed_sms= transform_text(input_sms)

    #2.vectorize
    vector_input= td.transform([transformed_sms])
    #3.predict

    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")






