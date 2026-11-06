"""Abstract base class defining the contract for all job scrapers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from scrapers.models import RawJobListing


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
