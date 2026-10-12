"""Define request and response schemas for the legal assistant API."""

from typing import List

from pydantic import BaseModel, Field, field_validator
from uuid import UUID


class QueryRequest(BaseModel):
    """Request body for a legal question."""

    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


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
    conversation_id: str | None = None
    cached: bool = False


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):
        import re
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address")
        return value


class Registration(Credentials):
    password: str = Field(min_length=4, max_length=128)


class HealthResponse(BaseModel):
    """API health status and retrieval index readiness."""

    status: str
    index_loaded: bool


class ErrorResponse(BaseModel):
    """Structured API error response."""

    error: str

