"""Define request and response schemas for the legal assistant API."""

from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for a legal question."""

    question: str = Field(..., min_length=1)


class Source(BaseModel):
    """Retrieved legal source returned with an answer."""
    act: str
    section_number: str
    section_title: str


class QueryResponse(BaseModel):
    """Generated legal answer and retrieved sources."""

    answer: str
    sources: List[Source]
    # The frontend must be able to tell "the law does not cover this" apart
    # from "the answer service is down" - without this the second silently
    # looks like the first.
    degraded: bool = False
    notice: str = ""


class HealthResponse(BaseModel):
    """API health status and retrieval index readiness."""

    status: str
    index_loaded: bool


class ErrorResponse(BaseModel):
    """Structured API error response."""

    error: str

