"""Prompt templates for Week 1 CompIntel flows."""

from __future__ import annotations

INTENT_ANALYST_SYSTEM_PROMPT = """You are CompIntel's intent analyst.
Extract the target company, market segment, competitors, and research questions.
Return strict JSON only.
"""


def build_intent_analyst_prompt(query: str) -> str:
    return f"{INTENT_ANALYST_SYSTEM_PROMPT}\nQuery: {query}"
