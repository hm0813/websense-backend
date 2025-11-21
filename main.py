from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import chromadb
from typing import List

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
# Embedding Model
# ------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ------------------------------------------------------------
# ChromaDB Setup (stable + simple)
# ------------------------------------------------------------

chroma_client = chromadb.Client()

def reset_collection():
    """Recreate collection fresh each search."""
    try:
        chroma_client.delete_collection("html_chunks")
    except:
        pass

    return chroma_client.get_or_create_collection(
        name="html_chunks",
        metadata={"hnsw:space": "cosine"}
    )

collection = reset_collection()


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def fetch_html(url: str) -> str:
    """Download HTML from the webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch website: {e}")


def extract_clean_text(html: str) -> str:
    """Clean page: remove scripts, styles, noscript, etc."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if len(t) > 20:
            parts.append(t)

    return "\n".join(parts)


def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    """Split cleaned text into chunks of max 500 tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def index_chunks(url: str, chunks: List[str]):
    """Stores chunks in vector DB."""
    global collection
    collection = reset_collection()

    if not chunks:
        return

    parsed = urlparse(url)
    path = parsed.path or "/"
    embeddings = embed_model.encode(chunks).tolist()

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"url": url, "path": path} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def semantic_search(query: str, k: int = 50):
    """Perform semantic search + sort by relevance + return top 10."""
    vec = embed_model.encode(query).tolist()

    result = collection.query(
        query_embeddings=[vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    if not result["documents"]:
        return []

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    ranked = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 / (1 + float(dist))  # convert distance → relevance score

        ranked.append({
            "path": meta.get("path", "/"),
            "title": doc[:120] + "...",
            "html": doc,
            "score": score,
        })

    # ⭐ Sort by descending score
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # ⭐ Return top 10
    return ranked[:10]


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.post("/search")
def search(req: SearchRequest):
    html = fetch_html(req.url)
    clean = extract_clean_text(html)
    chunks = chunk_text(clean)

    index_chunks(req.url, chunks)

    results = semantic_search(req.query)
    return {"results": results}


@app.get("/")
def root():
    return {"message": "Backend running successfully 🚀"}
