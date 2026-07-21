"""
trace.py

Creates the JSON payload returned by POST /ask.

This file DOES NOT modify MetaRAG.

It simply aggregates the public SDK methods into one
frontend-friendly response.
"""

from __future__ import annotations

import time
from typing import Dict, Any

from engine import (
    get_rag,
    analyze_query,
)

rag = get_rag()


# ==========================================================
# Helpers
# ==========================================================

def _build_query_section(query: str):

    """
    Query analysis + router prediction.
    """

    info = analyze_query(query)

    return {
        "query": query,
        "analysis": info["analysis"],
        "features": info["features"],
        "router": info["router"],
    }


def _build_answer_section(answer):

    """
    Converts MetaRAG Answer object
    into JSON.
    """

    return {

        "text": answer.text,

        "pipeline": answer.pipeline,

        "score": answer.score,

        "latency_ms": answer.latency_ms,

        "sources": answer.sources,

    }


def _build_trace_section(
    query: str,
    pipeline: str,
):

    """
    Uses SDK trace().
    """

    try:

        trace = rag.trace(
            query,
            pipeline_name=pipeline,
        )

    except Exception:

        trace = []

    return trace


def _build_inspection(query: str):

    """
    Compare every retriever.
    """

    try:

        inspection = rag.inspect(
            query,
            k=3,
        )

    except Exception:

        inspection = {}

    return inspection


def _build_explanation(query: str):

    """
    Router explanation.
    """

    try:

        explanation = rag.explain(query)

    except Exception:

        explanation = {}

    return explanation


def _build_corpus():

    """
    Corpus statistics.
    """

    try:

        return rag.analyze_corpus()

    except Exception:

        return {}


def _build_status():

    try:

        return rag.status()

    except Exception:

        return {}


# ==========================================================
# Public API
# ==========================================================

def build_trace(query: str) -> Dict[str, Any]:

    """
    Main backend entrypoint.

    Called by:

        POST /ask
    """

    total_start = time.perf_counter()

    # ---------------------------------------------
    # Query profiling
    # ---------------------------------------------

    query_section = _build_query_section(query)

    # ---------------------------------------------
    # Ask MetaRAG
    # ---------------------------------------------

    answer = rag.ask(query)

    # ---------------------------------------------
    # Trace
    # ---------------------------------------------

    trace = _build_trace_section(

        query,

        answer.pipeline,

    )

    # ---------------------------------------------
    # Retriever comparison
    # ---------------------------------------------

    inspection = _build_inspection(query)

    # ---------------------------------------------
    # Router explanation
    # ---------------------------------------------

    explanation = _build_explanation(query)

    # ---------------------------------------------
    # Corpus
    # ---------------------------------------------

    corpus = _build_corpus()

    # ---------------------------------------------
    # Status
    # ---------------------------------------------

    status = _build_status()

    total_latency = round(

        (time.perf_counter() - total_start) * 1000,

        2,

    )

    # =====================================================
    # Final payload
    # =====================================================

    return {

        "success": True,

        "query": query,

        "query_profile": query_section,

        "answer": _build_answer_section(answer),

        "trace": trace,

        "retrieval": inspection,

        "router": explanation,

        "corpus": corpus,

        "status": status,

        "timing": {

            "sdk_latency_ms": answer.latency_ms,

            "backend_latency_ms": total_latency,

        },

    }