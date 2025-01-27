import os
import sqlite3
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_fallback_secret")
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

@app.route("/process_resumes", methods=["POST"])
def process_resumes():
    job_description = request.json.get("job_description")
    resumes = request.json.get("resumes", [])

    if not job_description or not resumes:
        return jsonify({"error": "Job description and resumes are required"}), 400

    try:
        resume_texts = [extract_text_from_pdf(resume['file']) for resume in resumes]
        summaries = [summarize_text(text) for text in resume_texts]
        scores = rank_resumes(job_description, resume_texts)
        store_data_in_db(job_description, resume_texts, scores, summaries)
        return jsonify({"results": list(zip([resume['name'] for resume in resumes], scores, summaries))}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
