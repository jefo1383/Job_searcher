"""Data models and enumerations for job persistence."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle status of a scraped job offer.

    Attributes:
        NEW: Newly scraped offer, pending LLM scoring.
        PENDING_REVIEW: Matched offer waiting for human-in-the-loop validation.
        ACCEPTED: Approved by human, queued for application.
        REJECTED: Discarded either by LLM filter or human choice.
        APPLIED: Application successfully submitted.
    """

    NEW = "NEW"
    PENDING_REVIEW = "PENDING_REVIEW"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    APPLIED = "APPLIED"


class JobRecord(BaseModel):
    """Pydantic representation of a persisted job offer.

    Attributes:
        id: Primary key auto-incremented in SQLite (None for unpersisted jobs).
        platform: Originating portal name (e.g., 'linkedin', 'hellowork').
        company: Employing company or organization name.
        title: Position or role title.
        url: Original scraped job URL.
        normalized_url: Canonical sanitized URL for strict deduplication.
        description: Full text content of the job offer.
        raw_payload: Optional raw scrap data serialized as dictionary.
        score: Relevance score assigned by LLM (0-100, None if unevaluated).
        status: Current processing state.
        created_at: UTC timestamp of initial insertion.
        updated_at: UTC timestamp of last modification.
    """

    id: int | None = Field(default=None, description="SQLite primary key.")
    platform: str = Field(..., description="Source job portal identifier.")
    company: str = Field(..., description="Hiring company name.")
    title: str = Field(..., description="Job position title.")
    url: str = Field(..., description="Original raw URL.")
    normalized_url: str = Field(
        ...,
        description="Canonical URL used as unique deduplication key.",
    )
    description: str = Field(..., description="Full job description body.")
    raw_payload: dict[str, Any] | None = Field(
        default=None,
        description="Optional raw payload from scraper.",
    )
    score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Match score calculated by Gemini Flash.",
    )
    status: JobStatus = Field(
        default=JobStatus.NEW,
        description="Current processing lifecycle status.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp in UTC.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp in UTC.",
    )
