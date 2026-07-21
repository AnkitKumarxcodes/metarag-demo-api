"""
engine.py
---------

Creates and owns the singleton MetaRAG instance used by the API.

Nothing in this file is FastAPI-specific.
The API simply imports:

    from engine import rag

and uses the already-initialized object.

This file intentionally DOES NOT modify the MetaRAG SDK.
It simply wraps the published SDK.
"""

from __future__ import annotations

import os
import pickle
import joblib
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv

from google import genai
from sentence_transformers import SentenceTransformer


from metarag import (
    MetaRAG,
    CachedEmbeddings,
)

from metarag.router.router_interface import RouterInterface
from metarag.pipelines.generator import GeneratorInterface


# ============================================================
# Load Environment
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Create backend/.env containing:\n\n"
        "GEMINI_API_KEY=YOUR_KEY"
    )


# ============================================================
# Gemini Generator
# ============================================================

class GeminiGenerator(GeneratorInterface):

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-latest",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text


# ============================================================
# Router
# ============================================================

ROUTER_MODEL_PATH = BASE_DIR / "router.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

router_model = pickle.load(open(ROUTER_MODEL_PATH, "rb"))
label_encoder = joblib.load(LABEL_ENCODER_PATH)


class XGBRouter(RouterInterface):

    """
    Wrapper around your trained XGBoost router.

    Besides route(), we also expose predict_details()
    so the frontend can visualize probabilities.
    """

    def route(self, features: dict) -> str:

        if features.get("ocr_ratio") is None:
            features["ocr_ratio"] = 0.0

        row = (
            pd.DataFrame([features])
            .reindex(
                columns=router_model.feature_names_in_,
                fill_value=0,
            )
        )

        # print(row.dtypes)
        # print(row.iloc[0].to_dict())

        prediction = router_model.predict(row)[0]

        pipeline = label_encoder.inverse_transform([prediction])[0]

        return pipeline

    def predict_details(self, features):

        # Temporary workaround for MetaRAG v0.3.4 OCR bug
        features["ocr_ratio"] = 0.0

        row = (
            pd.DataFrame([features])
            .reindex(
                columns=router_model.feature_names_in_,
                fill_value=0,
            )
        )

        row = row.apply(pd.to_numeric, errors="coerce").fillna(0)

        probs = router_model.predict_proba(row)[0]

        probability_table = []

        for pipeline, score in zip(label_encoder.classes_, probs):

            probability_table.append(
                {
                    "pipeline": pipeline,
                    "score": round(float(score), 4),
                }
            )

        probability_table.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        prediction = router_model.predict(row)[0]

        selected = label_encoder.inverse_transform(
            [prediction]
        )[0]

        return {
            "selected": selected,
            "probabilities": probability_table,
        }

    def explain(
        self,
        pipeline: str,
        features=None,
    ):

        return f"{pipeline} selected by custom XGBoost router."


# ============================================================
# Embeddings
# ============================================================

class SentenceTransformerEmbeddings:
    """
    Wrapper to make SentenceTransformer compatible with MetaRAG.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].tolist()


embeddings = CachedEmbeddings(
    SentenceTransformerEmbeddings(
        "BAAI/bge-small-en-v1.5"
    ),
    cache_dir=str(BASE_DIR / ".metarag" / "embeddings"),
)


# ============================================================
# Build MetaRAG
# ============================================================

rag = MetaRAG(

    docs=str(BASE_DIR / "docs"),

    embeddings=embeddings,

    generator=GeminiGenerator(
        api_key=GEMINI_API_KEY,
    ),

    project="metarag_demo",

)

rag.fit()

router = XGBRouter()

rag.set_router(router)


# ============================================================
# Helper Functions
# ============================================================

def get_rag() -> MetaRAG:
    """
    Returns the singleton MetaRAG instance.
    """

    return rag


def get_router() -> XGBRouter:
    """
    Returns the singleton router.

    Used by trace.py to expose probabilities.
    """

    return router


def analyze_query(query: str):

    """
    Combines SDK query analysis with
    router feature extraction.

    This gives the frontend enough
    information to visualize the
    routing process.
    """

    analysis = rag.analyze_query(query)

    features = rag._extract_query_features(query)

    routing = router.predict_details(features)

    return {
        "analysis": analysis,
        "features": features,
        "router": routing,
    }

