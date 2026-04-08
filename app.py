from flask import Flask, request, jsonify, send_file
import requests
import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.util import ngrams
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import tempfile
import os
from dotenv import load_dotenv

# Load .env in local dev; Render will use environment variables
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Set it in .env or Render environment variables.")

app = Flask(__name__)

nltk.download("punkt", quiet=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN)

MIN_PHRASE_FREQ = 2
MIN_PHRASE_LEN = 12
NGRAM_RANGE = (2, 4)


def fetch_clean_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=15).text
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def extract_phrases(text, min_n=2, max_n=4):
    tokens = re.findall(r"[a-zA-Z]+", text)
    phrases = []

    for n in range(min_n, max_n + 1):
        phrases.extend([" ".join(g) for g in ngrams(tokens, n)])

    return phrases


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    urls = data.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    all_phrases = []

    try:
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
            return jsonify({"error": "No keywords found"}), 400

        embeddings = model.encode(keywords, convert_to_tensor=True)
        cosine_matrix = util.cos_sim(embeddings, embeddings)

        df = pd.DataFrame(
            cosine_matrix.cpu().numpy(),
            index=keywords,
            columns=keywords
        )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        df.to_excel(temp_file.name, index=True)

        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name="keyword_similarity.xlsx"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)