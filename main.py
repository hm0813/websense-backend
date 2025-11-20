from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# -----------------------
# Lightweight Embedding Model (Render-friendly)
# -----------------------
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def embed_text(text: str):
    """Create lightweight embedding using mean pooling."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)

    embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()


# -----------------------
# ChromaDB
# -----------------------
import chromadb
from typing import List

chroma_client = chromadb.Client()


def reset_collection():
    try:
        chroma_client.delete_collection("html_chunks")
    except:
        pass

    return chroma_client.get_or_create_collection(
        name="html_chunks",
        metadata={"hnsw:space": "cosine"}
    )


collection = reset_collection()

# -----------------------
# FastAPI Setup
# -----------------------

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


# -----------------------
# Helper Functions
# -----------------------

def fetch_html(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch website: {e}")


def extract_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if len(text) > 20:
            parts.append(text)

    return "\n".join(parts)


def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk_decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_decoded:
            chunks.append(chunk_decoded)

    return chunks


def index_chunks(url: str, chunks: List[str]):
    global collection
    collection = reset_collection()

    if not chunks:
        return

    parsed = urlparse(url)
    path = parsed.path or "/"

    embeddings = [embed_text(chunk) for chunk in chunks]

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"url": url, "path": path} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def semantic_search(query: str, k: int = 50):
    query_vec = embed_text(query)

    result = collection.query(
        query_embeddings=[query_vec],
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
        score = 1 / (1 + float(dist))

        ranked.append({
            "path": meta.get("path", "/"),
            "title": doc[:120] + "...",
            "html": doc,
            "score": score,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:10]


# -----------------------
# Routes
# -----------------------

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
