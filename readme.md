# MetaRAG Playground Backend

A FastAPI backend that exposes the MetaRAG SDK through a REST API for interactive RAG pipeline analysis, routing, retrieval inspection, and answer generation.

This project demonstrates how MetaRAG can be integrated into an application while keeping the SDK unchanged.

---

## Features

- REST API built with FastAPI
- Automatic RAG pipeline selection using a custom XGBoost router
- Query analysis and feature extraction
- Retrieval inspection across multiple pipelines
- Pipeline execution tracing
- Gemini-powered answer generation
- SentenceTransformer embeddings
- Interactive Swagger documentation

---

## Architecture

```
                User
                  │
                  ▼
             FastAPI API
                  │
        ┌─────────┴─────────┐
        │                   │
   MetaRAG SDK         XGBoost Router
        │                   │
        ├──────────────┐
        │              │
 Retrieval        Gemini Generator
        │              │
 SentenceTransformer Embeddings
```

---

## API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Service information |
| GET | `/health` | Health check |
| GET | `/status` | MetaRAG status |
| GET | `/pipelines` | Available pipelines |
| POST | `/ask` | Execute a query |
| GET | `/docs` | Interactive Swagger UI |

---

## Installation

### Clone

```bash
git clone https://github.com/<username>/metarag-playground-backend.git

cd metarag-playground-backend
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Documents

Place your knowledge base inside

```
docs/
```

Example

```
backend/
│
├── docs/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
```

---

## Running

```bash
uvicorn main:app --reload
```

Server

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

---

## Example Request

```http
POST /ask
```

```json
{
  "query": "Explain hybrid retrieval."
}
```

Example response

```json
{
  "success": true,
  "answer": {
    "text": "...",
    "pipeline": "hybrid",
    "score": 0.92
  }
}
```

---

## Tech Stack

- FastAPI
- MetaRAG SDK
- XGBoost
- SentenceTransformers
- Google Gemini
- Pandas
- Scikit-learn

---

## Project Structure

```
backend/
├── docs/
├── engine.py
├── trace.py
├── main.py
├── router.pkl
├── label_encoder.pkl
├── requirements.txt
└── .env
```

---

## Notes

- This project is a demonstration backend built on top of the MetaRAG SDK.
- The MetaRAG SDK itself is not modified.
- Routing decisions are produced by a custom XGBoost classifier trained for pipeline selection.

---

## License

MIT