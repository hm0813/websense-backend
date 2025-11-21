
#  **websense-backend — Semantic Search Backend**

A FastAPI-based backend service for extracting website HTML, chunking content, generating embeddings, and performing semantic search using ChromaDB.

---

## 🔧 **Features**

* Fetches & parses HTML from any given URL
* Splits content into 500-token chunks
* Generates embeddings using **Sentence-Transformers (MiniLM-L6-v2)**
* Stores vectors in **ChromaDB**
* Returns the **top 10 relevant matches** for a search query
* Clean, modular, and lightweight

---

## 🚀 **Tech Stack**

* **FastAPI**
* **Python 3.10+**
* **BeautifulSoup4**
* **Sentence-Transformers**
* **ChromaDB**
* **Requests**

---

## ▶️ **How to Run**

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Start FastAPI server

```
uvicorn main:app --reload
```

Server runs at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📌 **API Endpoints**

### **POST /search**

Input:

```json
{
  "url": "https://example.com",
  "query": "your search text"
}
```

Output:

* Top 10 matching HTML chunks with scores

---

## 📁 **Project Structure**

```
backend/
│── main.py
│── requirements.txt
│── chroma_db/
│── README.md
```



## ✨ Part of the WebSense Full-Stack Project

Frontend + main dashboard:
👉 [https://github.com/hm0813/websense-ai-semantic-search](https://github.com/hm0813/websense-ai-semantic-search)

---
