import time
import json
import mysql.connector
import tempfile
import os
import requests
import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.util import ngrams
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()
nltk.download("punkt", quiet=True)

DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")
PORT = os.environ.get("PORT")
HF_TOKEN = os.environ.get("HF_TOKEN")
LARAVEL_UPLOAD_URL = os.environ.get("LARAVEL_UPLOAD_URL")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN)

MIN_PHRASE_FREQ = 2
MIN_PHRASE_LEN = 12
NGRAM_RANGE = (2, 4)


def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=PORT,
    )


def fetch_clean_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=15).text
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).lower()


def extract_phrases(text, min_n=2, max_n=4):
    tokens = re.findall(r"[a-zA-Z]+", text)
    phrases = []
    for n in range(min_n, max_n + 1):
        phrases.extend([" ".join(g) for g in ngrams(tokens, n)])
    return phrases


def process_job(kwr):
    kwr_id = kwr["id"]
    urls = json.loads(kwr["urls"])
    print(f"Processing analysis request {kwr_id}")

    all_phrases = []
    for url in urls:
        text = fetch_clean_text(url)
        phrases = extract_phrases(text, *NGRAM_RANGE)
        all_phrases.extend(phrases)

    phrase_counts = Counter(all_phrases)
    keywords = [
        phrase for phrase, count in phrase_counts.items()
        if count >= MIN_PHRASE_FREQ and len(phrase) >= MIN_PHRASE_LEN
    ]

    if not keywords:
        raise Exception("No keywords found")

    embeddings = model.encode(keywords, convert_to_tensor=True)
    cosine_matrix = util.cos_sim(embeddings, embeddings)

    df = pd.DataFrame(
        cosine_matrix.cpu().numpy(),
        index=keywords,
        columns=keywords
    )

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"kwr_{kwr_id}.xlsx")
    df.to_excel(file_path, index=True)
    return file_path



def send_file_to_laravel(kwr_id, file_path):
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"kwr_id": kwr_id}
            response = requests.post(LARAVEL_UPLOAD_URL, files=files, data=data)
            if response.status_code == 200:
                res = response.json()
                print(f"Uploaded to Laravel: {res.get('file_path')}")
                return res.get("file_path")
            else:
                print("Failed to upload:", response.text)
                return None
    except Exception as e:
        print("Error sending file to Laravel:", e)
        return None


def update_job(job_id, status, output_file=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        UPDATE keyword_jobs
        SET status=%s, output_file=%s, updated_at=NOW()
        WHERE id=%s
    """
    cursor.execute(query, (status, output_file, job_id))
    conn.commit()
    cursor.close()
    conn.close()


def fetch_job():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT * FROM keyword_jobs
        WHERE status='pending'
        ORDER BY id ASC
        LIMIT 1
    """)
    job = cursor.fetchone()
    if job:
        cursor.execute("UPDATE keyword_jobs SET status='processing' WHERE id=%s", (job["id"],))
        conn.commit()
    cursor.close()
    conn.close()
    return job



while True:
    try:
        job = fetch_job()
        if job:
            try:
                file_path = process_job(job)
                uploaded_path = send_file_to_laravel(job["id"], file_path)
                if uploaded_path:
                    update_job(job["id"], "done", uploaded_path)
                else:
                    update_job(job["id"], "failed")
            except Exception as e:
                print("Error processing job:", e)
                update_job(job["id"], "failed")
        else:
            print("No jobs...")
        time.sleep(5)
    except Exception as e:
        print("Worker error:", e)
        time.sleep(10)