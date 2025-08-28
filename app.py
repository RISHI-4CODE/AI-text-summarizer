# app.py
import streamlit as st
from transformers import pipeline

# Load model once, cache it for performance
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# UI
st.title("📖 AI-Powered Text Summarizer")
st.write("Paste any long article and get a concise summary instantly!")

# Text input
text_input = st.text_area("Enter text to summarize:", height=200)

# Button action
if st.button("Summarize"):
    if text_input.strip():
        summary = summarizer(text_input, max_length=150, min_length=40, do_sample=False)
        st.subheader("Summary")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text.")
