"""Abstract base class defining the contract for all job scrapers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import re
import unicodedata

from scrapers.models import RawJobListing

# Mots-clés bloquants dans l'intitulé du poste (paramètre réglable)
DEFAULT_EXCLUDED_TITLE_KEYWORDS: tuple[str, ...] = (
    "senior",
    "manager",
    "head of",
    "directeur",
    "directrice",
    "director",
)


def is_excluded_title(
    title: str,
    excluded_keywords: tuple[str, ...] = DEFAULT_EXCLUDED_TITLE_KEYWORDS,
) -> bool:
    """Check whether a job title contains seniority or leadership blockers.

    Normalizes accents and matches whole words to prevent false positives.

    Args:
        title: Raw job position title.
        excluded_keywords: Tuple of blocked title tokens.

    Returns:
        bool: True if the title should be discarded before extraction/analysis.
    """
    if not title:
        return False
    decomposed = unicodedata.normalize("NFKD", title)
    normalized = "".join(
        c for c in decomposed if not unicodedata.combining(c)
    ).lower()
    for kw in excluded_keywords:
        parts = [re.escape(part) for part in kw.lower().split()]
        if not parts:
            continue
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
        if re.search(pattern, normalized):
            return True
    return False


class BaseScraper(ABC):
    """Abstract base scraper defining common interface across job platforms.

    Attributes:
        name: Unique identifier for the scraper platform.
    """

    def __init__(self, name: str) -> None:
        """Initialize the scraper.

        Args:
            name: Platform identifier name.
        """
        self.name = name

    @abstractmethod
    def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Asynchronously scrape job postings matching criteria.

        Args:
            keywords: List of target job title or skill keywords.
            locations: List of target geographical locations or remote tags.
            limit: Maximum count of postings to extract per execution.

        Yields:
            RawJobListing: Extracted raw job post item.
        """
        pass
