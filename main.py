from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------------------------------------
# Fast, lightweight CPU-only embedding model
# ------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")    # CPU model, ~90MB

# ------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    url: str
    query: str


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def fetch_html(url: str) -> str:
    """Download HTML safely."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return res.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch website: {e}")


def extract_clean_text(html: str):
    """Extract readable text from HTML (headlines, paragraphs, list items)."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if len(t) > 20:
            parts.append(t)

    return parts


def embed(texts):
    """Create embeddings in batch."""
    return model.encode(texts).tolist()


def semantic_search(chunks, query):
    """Compute cosine similarity manually (no ChromaDB)."""
    if not chunks:
        return []

    chunk_embeddings = np.array(embed(chunks))
    query_emb = np.array(embed([query])[0])

    # Cosine similarity
    similarities = chunk_embeddings @ query_emb / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    # Sort highest → lowest
    ranked = np.argsort(similarities)[::-1][:10]

    results = []
    for idx in ranked:
        results.append({
            "path": "/",
            "title": chunks[idx][:100] + "...",
            "html": chunks[idx],
            "score": float(similarities[idx])
        })

    return results


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.post("/search")
def search(req: SearchRequest):
    html = fetch_html(req.url)
    cleaned_chunks = extract_clean_text(html)

    results = semantic_search(cleaned_chunks, req.query)

    return {"results": results}


@app.get("/")
def root():
    return {"message": "Backend running successfully 🚀"}
