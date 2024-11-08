import openai
import streamlit as st
from transformers import pipeline

# Access the API keys from Streamlit's secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
huggingface_api_key = st.secrets["HUGGINGFACE_API_KEY"]

# Initialize Hugging Face summarization pipeline with API key for authentication
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=huggingface_api_key)

# Function for Text Summarization
def generate_summary(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to generate multiple-choice quiz questions from text
def generate_questions(text):
    prompt = f"Generate 5 multiple-choice questions based on the following text:\n\n{text}"
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.title("AI Text Tools")

# Create a navbar for navigation
option = st.sidebar.selectbox(
    'Choose a tool',
    ('Text Summarization', 'Quiz Generator')
)

# Text Summarization Tool
if option == 'Text Summarization':
    st.header("Text Summarization Tool")
    text_input = st.text_area("Enter Text to Summarize", height=200)
    
    if st.button("Summarize"):
        if text_input:
            summary = generate_summary(text_input)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

# Quiz Generator Tool
elif option == 'Quiz Generator':
    st.header("Quiz Generator Tool")
    text_input = st.text_area("Enter Text to Generate Quiz Questions", height=200)
    
    if st.button("Generate Quiz"):
        if text_input:
            questions = generate_questions(text_input)
            st.subheader("Generated Quiz Questions:")
            st.write(questions)
        else:
            st.warning("Please enter some text to generate quiz questions.")
