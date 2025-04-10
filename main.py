import nltk

import pandas as pd
import numpy as np
import faiss
import sqlite3
import re
from glob import glob

import spacy
import fitz

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# ----------------------------
# Config
# ----------------------------
DB_PATH = 'recruitment.db'
JD_CSV = './JD and Resume/job_description.csv'

# ----------------------------
# Load NLP Model
# ----------------------------
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# Load LLaMA & Embeddings
# ----------------------------
llm = OllamaLLM(model="tinydolphin", temperature=0.7, num_predict=500)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ----------------------------
# Summarizer + Embedder
# ----------------------------
def summarize_jd(text):
    prompt = f"Summarize the following job description:\n\n{text}\n\nSummary:"
    return llm.invoke(prompt).strip()

def get_embedding(text):
    return embeddings.embed_query(text)

def normalize(vec):
    return vec / np.linalg.norm(vec)

# ----------------------------
# Process Job Descriptions
# ----------------------------
def process_jds(csv_path):
    df = pd.read_csv(csv_path, encoding='cp1252')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    summary TEXT
                )''')

    vectors = []
    for _, row in df.iterrows():
        title, desc = row['Job Title'], row['Job Description']
        summary = summarize_jd(desc)
        vec = get_embedding(summary)
        normalized_vec = normalize(vec)   # Normalize the embedding
        vectors.append(normalized_vec)  
        c.execute("INSERT INTO job_descriptions (title, description, summary) VALUES (?, ?, ?)",
                  (title, desc, summary))

    conn.commit()
    conn.close()

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype('float32'))
    faiss.write_index(index, 'jd_index.faiss')

if __name__ == '__main__':
    process_jds(JD_CSV)
    
