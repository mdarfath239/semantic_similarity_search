# Semantic Similarity Search System

> An AI-powered semantic document search web application built with Word2Vec, Flask, and a professional responsive frontend.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [Project Architecture](#project-architecture)
6. [Model Information](#model-information)
7. [Installation](#installation)
8. [Running the Application](#running-the-application)
9. [API Documentation](#api-documentation)
10. [Example Search Queries](#example-search-queries)
11. [Testing](#testing)
12. [Screenshots](#screenshots)
13. [Limitations](#limitations)
14. [Future Improvements](#future-improvements)

---

## Project Overview

**Semantic Similarity Search** is a full-stack NLP application that retrieves documents from the **20 Newsgroups** corpus by understanding the *meaning* of a user's natural-language query — not just exact keywords.

The backend exposes a REST API powered by a pre-trained **Word2Vec** model. The frontend provides a clean, professional dashboard where users can search, filter results, visualise similarity scores, and review their search history.

---

## Problem Statement

Traditional keyword-based search fails when the user's query and the document use different words to express the same concept (e.g. "car" vs "automobile"). Semantic search addresses this by embedding both the query and each document into a shared vector space and ranking documents by their geometric proximity to the query.

---

## Features

| Feature | Description |
|---|---|
| Semantic search | Word2Vec embeddings + cosine similarity |
| Professional dashboard | Hero section, stats strip, result cards |
| Similarity score bars | Percentage progress bars per result |
| Expand / collapse text | Show full document text on demand |
| Copy text | Copy document content to clipboard |
| Category filter | Filter results by newsgroup category |
| Score threshold | Slider to hide low-confidence results |
| Search history | Stored in browser localStorage, click-to-reuse |
| Search suggestions | One-click example queries |
| Responsive design | Mobile, tablet, desktop |
| Health endpoint | `/api/health` reports model loading status |
| Input validation | Server-side and client-side guards |
| Error handling | User-friendly error messages |

---

## Technology Stack

### Backend
- **Python 3.10**
- **Flask 3** — web framework
- **Gensim** — Word2Vec model loading and inference
- **NumPy / Pandas** — numerical operations
- **scikit-learn** — cosine similarity
- **NLTK** — text tokenisation and stopword removal
- **Joblib** — model artifact deserialisation

### Frontend
- **HTML5 / CSS3 / JavaScript (ES2022)**
- **Bootstrap 5.3** — responsive layout
- **Font Awesome 6** — icons
- **Google Fonts: Inter** — typography

---

## Project Architecture

```text
semantic_similarity_search/
│
├── models/                        ← Pre-trained model artifacts (not modified)
│   ├── word2vec_model.model
│   ├── search_document_vectors.pkl
│   └── search_document_metadata.pkl
│
├── src/                           ← Existing NLP inference layer (not modified)
│   ├── preprocessing.py           ← Text cleaning & tokenisation
│   └── inference.py               ← semantic_search() function
│
├── app/                           ← Flask web application layer
│   ├── __init__.py                ← Application factory (create_app)
│   ├── routes.py                  ← HTTP route handlers (thin controllers)
│   └── services.py                ← Business logic, validation, timing
│
├── templates/                     ← Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   └── about.html
│
├── static/
│   ├── css/style.css              ← Custom stylesheet
│   └── js/app.js                  ← Frontend JavaScript
│
├── tests/
│   ├── test_routes.py             ← Unit tests (mocked model)
│   └── test_search.py             ← Integration tests (real model)
│
├── app.py                         ← Server entry point
├── requirements.txt
├── .gitignore
└── README.md
```

### Request flow

```
Browser → GET /            → Flask renders index.html
Browser → POST /api/search → routes.py → services.py → src/inference.py → JSON response
Browser → GET /api/health  → routes.py → services.py → JSON status
```

---

## Model Information

| Property | Value |
|---|---|
| Algorithm | Word2Vec |
| Embedding dimensions | 100 |
| Similarity metric | Cosine Similarity |
| Dataset | 20 Newsgroups |
| Total indexed documents | 18 227 |
| Library | Gensim |
| Training environment | Google Colab |

The model was trained externally and is loaded once at application startup. It is **never retrained** by the Flask application.

### Dataset: 20 Newsgroups

The [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/) contains approximately 18 000 newsgroup posts across 20 categories:

`alt.atheism` · `comp.graphics` · `comp.os.ms-windows.misc` · `comp.sys.ibm.pc.hardware` · `comp.sys.mac.hardware` · `comp.windows.x` · `misc.forsale` · `rec.autos` · `rec.motorcycles` · `rec.sport.baseball` · `rec.sport.hockey` · `sci.crypt` · `sci.electronics` · `sci.med` · `sci.space` · `soc.religion.christian` · `talk.politics.guns` · `talk.politics.mideast` · `talk.politics.misc` · `talk.religion.misc`

---

## Installation

### Prerequisites

- Python 3.10+
- The pre-trained model files in `models/`

### 1. Clone / navigate to the project

```powershell
cd C:\path\to\semantic_similarity_search
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note**: If you receive a `LookupError` about NLTK data on first run, the application will auto-download the required `stopwords` corpus.

---

## Running the Application

```powershell
# From the project root with the virtual environment activated:
.venv\Scripts\python.exe app.py
```

Then open your browser at: **http://127.0.0.1:5000**

### Environment variables (optional)

| Variable | Default | Description |
|---|---|---|
| `FLASK_HOST` | `127.0.0.1` | Bind address |
| `FLASK_PORT` | `5000` | Bind port |
| `FLASK_DEBUG` | `false` | Enable debug mode |

```powershell
$env:FLASK_DEBUG = "true"
.venv\Scripts\python.exe app.py
```

---

## API Documentation

### `GET /`
Renders the main search dashboard.

---

### `GET /about`
Renders the About / Model Info page.

---

### `GET /api/health`

Returns application and model health.

**Response (200 — healthy):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_name": "Word2Vec",
    "embedding_dimensions": 100,
    "total_indexed_documents": 18227,
    "dataset": "20 Newsgroups",
    "similarity_metric": "Cosine Similarity"
  }
}
```

**Response (503 — degraded):**
```json
{
  "status": "degraded",
  "model_loaded": false,
  "error": "Model file not found: ..."
}
```

---

### `POST /api/search`

Performs a semantic similarity search.

**Request body:**
```json
{
  "query": "computer graphics and image processing",
  "top_n": 5
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query` | string | Yes | 1–2000 characters |
| `top_n` | integer | No (default: 5) | 1–20 |

**Response (200 — success):**
```json
{
  "success": true,
  "query": "computer graphics and image processing",
  "total_results": 5,
  "execution_time_ms": 14.2,
  "results": [
    {
      "rank": 1,
      "document_id": "comp.graphics_12345",
      "category": "comp.graphics",
      "similarity_score": 0.87,
      "text": "..."
    }
  ]
}
```

**Response (400 — validation error):**
```json
{
  "success": false,
  "error": "Query must be a non-empty string."
}
```

---

## Example Search Queries

| Query | Expected category match |
|---|---|
| `computer graphics and image processing` | `comp.graphics` |
| `space exploration and NASA missions` | `sci.space` |
| `religion and ethics in society` | `soc.religion.christian` / `alt.atheism` |
| `encryption and internet security` | `sci.crypt` |
| `electric vehicles and energy` | `sci.electronics` / `rec.autos` |
| `baseball season statistics` | `rec.sport.baseball` |
| `gun control legislation` | `talk.politics.guns` |
| `medical diagnosis and treatment` | `sci.med` |

---

## Testing

### Unit tests (mocked model — fast)

```powershell
.venv\Scripts\python.exe -m pytest tests\test_routes.py -v
```

### Integration tests (real model — requires model files)

```powershell
.venv\Scripts\python.exe -m pytest tests\test_search.py -v
```

### All tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

---

## Screenshots

> _Add screenshots of the running application here._

| View | Description |
|---|---|
| `screenshots/home.png` | Main search dashboard |
| `screenshots/results.png` | Search results with similarity bars |
| `screenshots/about.png` | About / model info page |
| `screenshots/mobile.png` | Mobile responsive view |

---

## Limitations

- Queries containing words **absent from the Word2Vec vocabulary** return no results.
- The model is trained only on **20 Newsgroups** — out-of-domain queries (e.g. medical or legal text) may produce low-quality results.
- **Word averaging** loses word-order and negation information.
- Cosine similarity is a geometric measure, not a relevance probability.
- The application is a **portfolio demonstration** and is not hardened for public production deployment (no rate limiting, authentication, or HTTPS).

---

## Future Improvements

- [ ] Replace word-averaging with **sentence-level embeddings** (e.g. SBERT, OpenAI Embeddings)
- [ ] Add **TF-IDF hybrid scoring** to boost term-specific results
- [ ] Implement **approximate nearest-neighbour** search (FAISS, Annoy) for faster retrieval at scale
- [ ] Add **pagination** for large result sets
- [ ] Support uploading a **custom document corpus**
- [ ] Add **authentication** and usage tracking
- [ ] Deploy with **Gunicorn + Nginx** behind HTTPS
- [ ] Add **Docker** containerisation
