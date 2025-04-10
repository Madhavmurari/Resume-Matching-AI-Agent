import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import fitz
import faiss
import spacy

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# ----------------------------
# Config
# ----------------------------
DB_PATH = 'recruitment.db'
FAISS_INDEX = 'jd_index.faiss'

# Load models
nlp = spacy.load("en_core_web_sm")
llm = OllamaLLM(model="tinydolphin", temperature=0.7, num_predict=500)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ----------------------------
# Utility Functions
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_embedding(text):
    return embeddings.embed_query(text)

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def summarize_resume(text):
    prompt = f"Summarize this resume in a few concise bullet points:\n{text}"
    summary = llm.invoke(prompt)
    return summary

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("AI Resume Matcher & Interview Scheduler")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.write("---")
    st.subheader("Extracted Resume Text:")
    text = extract_text_from_pdf(uploaded_file)
    st.text_area("Resume", text, height=300)

    # Extract name and email
    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
    email = next((ent.text for ent in doc.ents if ent.label_ == "EMAIL"), "Not found")

    # Summarize resume before embedding
    resume_summary = summarize_resume(text)
    st.subheader("Resume Summary:")
    st.text_area("Summary", resume_summary, height=150)

    resume_embedding = get_embedding(resume_summary)
    normalized_resume_embedding = normalize(resume_embedding)

    # Reuse DB connection
    with sqlite3.connect(DB_PATH) as conn:
        # Load job descriptions and FAISS index
        jd_df = pd.read_sql_query("SELECT * FROM job_descriptions", conn)
        index = faiss.read_index(FAISS_INDEX)

        # Perform similarity search
        D, I = index.search(np.array([normalized_resume_embedding]).astype('float32'), k=3)

        st.subheader("\U0001F4C4 Job Description Match Results:")
        shortlisted = False

        for i, dist in zip(I[0], D[0]):
            job = jd_df.iloc[i]
            dist=dist*25
            score = 100 - dist
            st.markdown(f"**Job Title:** {job['title']}")
            st.markdown(f"**Summary:** {job['summary']}")
            st.markdown(f"**Match Score:** {score:.2f}%")
            st.markdown("---")

            if score >= 80 and not shortlisted:
                shortlisted = True
                st.success("\u2705 Candidate shortlisted for interview!")

                interview_date = "2025-04-10"
                interview_time = "10:00 AM"
                interview_format = "Zoom"

                c = conn.cursor()

                # Create candidates table if it doesn't exist
                c.execute('''CREATE TABLE IF NOT EXISTS candidates (
                                id INTEGER PRIMARY KEY,
                                name TEXT,
                                email TEXT,
                                cv_path TEXT,
                                match_score REAL,
                                shortlisted INTEGER DEFAULT 0
                            )''')

                c.execute("INSERT INTO candidates (name, email, cv_path, match_score, shortlisted) VALUES (?, ?, ?, ?, ?)",
                          (name, email, uploaded_file.name, score, 1))
                candidate_id = c.lastrowid

                # Create interviews table if it doesn't exist
                c.execute('''CREATE TABLE IF NOT EXISTS interviews (
                                candidate_id INTEGER,
                                interview_date TEXT,
                                time TEXT,
                                format TEXT
                            )''')

                c.execute("INSERT INTO interviews (candidate_id, interview_date, time, format) VALUES (?, ?, ?, ?)",
                          (candidate_id, interview_date, interview_time, interview_format))

                conn.commit()

                st.markdown(f"**Interview Scheduled on {interview_date} at {interview_time} via {interview_format}**")
                break

        if not shortlisted:
            st.warning("\u274C No suitable match (score below 80%). Not shortlisted.")
