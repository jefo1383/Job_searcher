"""Data models for raw scraped job listings before persistence."""

from typing import Any
from pydantic import BaseModel, Field


class RawJobListing(BaseModel):
    """Raw job post extracted by a scraper before database ingestion.

    Attributes:
        platform: Identifier of the source portal (e.g. 'wttj', 'linkedin').
        company: Name of the hiring company.
        title: Position or role title.
        url: Original target URL of the listing.
        description: Textual content of the job offer.
        location: Location string (city, region, remote).
        raw_payload: Optional raw dictionary received from source API or parser.
    """

    platform: str = Field(..., description="Originating job board identifier.")
    company: str = Field(..., description="Name of employing company.")
    title: str = Field(..., description="Title of the job posting.")
    url: str = Field(..., description="Original URL.")
    description: str = Field(..., description="Full text description of offer.")
    location: str = Field(default="", description="Location or remote policy.")
    raw_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw data payload from source.",
    )
