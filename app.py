import os
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import secrets  # Import secrets to generate a secret key

load_dotenv()

app = Flask(__name__)

# Generate a secret key dynamically
app.secret_key = secrets.token_hex(24)  # This will generate a random 48-character hexadecimal string

UPLOAD_FOLDER = 'static/resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Hardcoded credentials for login
USERNAME = "admin"
PASSWORD = "admin"

# Set up Google API key for ChatGoogleGenerativeAI
google_api_key = os.getenv("google_api_key")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to summarize text using ChatGoogleGenerativeAI
def summarize_text(text):
    try:
        # Initialize ChatGoogleGenerativeAI instance
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
        
        # Format the prompt for summarization
        prompt = f"Summarize the following pdf resumes: {text[:1500]}"  # Limit to first 1500 characters for API call

        # Get response from the model
        response = llm.invoke(prompt)
        
        # Access the content of the response correctly
        summary = response.content.strip()  # Use .content to get the message content
        
        return summary
    except Exception as e:
        return f"Error in summarization: {e}"

# Function to store data in SQLite3
def store_data_in_db(job_description, resumes, scores, summaries):
    try:
        conn = sqlite3.connect('resumes.db')
        c = conn.cursor()
        c.execute(''' 
        CREATE TABLE IF NOT EXISTS resume_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            job_description TEXT, 
            resume_name TEXT, 
            resume_text TEXT, 
            score REAL, 
            summary TEXT
        ) ''')

        for i, resume_text in enumerate(resumes):
            c.execute('''
            INSERT INTO resume_data (job_description, resume_name, resume_text, score, summary)
            VALUES (?, ?, ?, ?, ?) 
            ''', (job_description, f"resume_{i + 1}", resume_text, scores[i], summaries[i]))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# Streamlit login route
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            st.session_state.logged_in = True
            st.experimental_rerun()  # Refresh the page after login
        else:
            st.error("Invalid username or password")

# Streamlit main route
def main():
    if not st.session_state.get("logged_in", False):
        login()
    else:
        st.title("HR Resume Screening Tool")

        job_description = st.text_area("Job Description")
        uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf"])

        if st.button("Process Resumes"):
            if not job_description:
                st.error("Job description is required.")
            elif not uploaded_files:
                st.error("Please upload at least one resume.")
            else:
                resumes = []
                summaries = []

                for file in uploaded_files:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.name)
                    with open(filepath, "wb") as f:
                        f.write(file.getbuffer())

                    resume_text = extract_text_from_pdf(filepath)
                    if resume_text.startswith("Error"):
                        continue

                    resumes.append(resume_text)
                    summaries.append(summarize_text(resume_text))

                scores = rank_resumes(job_description, resumes)
                store_data_in_db(job_description, resumes, scores, summaries)

                scores = [round(score, 2) for score in scores]

                results = list(zip([file.name for file in uploaded_files], scores, summaries))
                results.sort(key=lambda x: x[1], reverse=True)

                if results:
                    top_match = results[0]
                    st.write("Top Match:")
                    st.write(top_match)
                    st.write("All Results:")
                    for result in results:
                        st.write(result)
        
        # Logout button
        if st.button("Logout"):
            st.session_state.clear()  # Clear session state for logout
            st.experimental_rerun()

# Main logic to run Streamlit app
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

main()
