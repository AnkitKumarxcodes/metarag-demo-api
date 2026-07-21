"""
main.py
-------

FastAPI entrypoint for the MetaRAG Playground.

Responsibilities
----------------
- expose REST endpoints
- validate requests
- return JSON

Business logic lives in trace.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import get_rag
from trace import build_trace

rag = get_rag()
# ==========================================================
# FastAPI
# ==========================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n" + "=" * 70)
    print("🚀 MetaRAG Playground Backend")
    print("=" * 70)
    print(f"Project     : {rag.project}")
    print(f"Pipelines   : {len(rag._pipelines)}")
    print(f"Available   : {', '.join(rag._pipelines.keys())}")
    print(f"Router      : {rag.status()['router']}")
    print(f"Docs        : http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")

    yield

app = FastAPI(
    title="MetaRAG Playground",
    version="0.3.5",
    description="Interactive visualization for MetaRAG.",
    lifespan=lifespan,
)

# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# ==========================================================
# Models
# ==========================================================

class AskRequest(BaseModel):
    query: str


# ==========================================================
# Routes
# ==========================================================

@app.get("/")
def root():

    return {
        "name": "MetaRAG Playground",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


@app.get("/status")
def status():

    return rag.status()


@app.get("/pipelines")
def pipelines():

    return {
        "pipelines": list(rag._pipelines.keys())
    }


@app.post("/ask")
def ask(request: AskRequest):

    query = request.query.strip()

    if not query:
        return {
            "success": False,
            "error": "Query cannot be empty."
        }

    return build_trace(query)