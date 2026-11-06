"""Scraper implementation for Orange Cyberdefense career opportunities."""

import html
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote_plus

import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# En-têtes HTTP simulant une requête de navigation standard
DEFAULT_ORANGE_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Racine du portail d'offres du groupe Orange
ORANGE_JOBS_API_URL: str = "https://orange.jobs/jobs/v3/offers"
ORANGE_PUBLIC_BASE_URL: str = "https://orange.jobs/jobs/offer"


def _clean_html(raw_html: str | None) -> str:
    """Strip HTML markup and collapse superfluous whitespace.

    Args:
        raw_html: Raw HTML string.

    Returns:
        str: Sanitized plain text string.
    """
    if not raw_html:
        return ""
    unescaped = html.unescape(raw_html)
    clean_text = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", clean_text).strip()


class OrangeCyberScraper(BaseScraper):
    """Scraper targeting Orange Cyberdefense positions via Orange Jobs REST API.

    Attributes:
        company_name: Hiring organization identifier.
        entity: Specific Orange subsidiary entity to filter.
        allowed_locations: Whitelist of allowed location substrings.
        timeout_seconds: Request timeout duration in seconds.
    """

    def __init__(
        self,
        company_name: str = "Orange Cyberdefense",
        entity: str = "Orange Cyberdefense",
        allowed_locations: list[str] | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        """Initialize the Orange Cyberdefense scraper.

        Args:
            company_name: Name of hiring company.
            entity: Filter segment for target Orange business unit.
            allowed_locations: List of target location tokens.
            timeout_seconds: Network call timeout in seconds.
        """
        super().__init__(name="orange_cyberdefense")
        self.company_name = company_name
        self.entity = entity
        self.allowed_locations = [
            loc.lower()
            for loc in (
                allowed_locations
                or ["marseille", "aix", "13", "remote", "télétravail"]
            )
        ]
        self.timeout_seconds = timeout_seconds

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Query Orange jobs API for matching positions.

        Args:
            keywords: Target role or technology keywords.
            locations: Secondary geographic search criteria.
            limit: Maximum count of listings to extract per search.

        Yields:
            RawJobListing: Normalized scraped job post.
        """
        async with httpx.AsyncClient(
            headers=DEFAULT_ORANGE_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for keyword in keywords:
                # Paramètres de recherche ciblant la France et l'entité cyber
                params: dict[str, str | int] = {
                    "keyword": keyword,
                    "lang": "FR",
                    "country": "France",
                    "limit": min(limit, 20),
                }

                try:
                    response = await client.get(
                        ORANGE_JOBS_API_URL,
                        params=params,
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    offers = self._extract_offers_list(data)

                    for item in offers:
                        title = item.get("title", "").strip()
                        raw_loc = self._extract_location(item)
                        offer_id = str(item.get("id", "")).strip()

                        if not title or not offer_id:
                            continue

                        # Vérification de l'entité (doit être cyber ou
                        # mentionnée dans le titre/mot-clé)
                        item_entity = item.get("entity", "")
                        if (
                            self.entity
                            and self.entity.lower()
                            not in item_entity.lower()
                            and "cyber" not in title.lower()
                            and "cyber" not in keyword.lower()
                        ):
                            continue

                        # Vérification de la zone géographique
                        if not self._is_location_allowed(raw_loc):
                            continue

                        public_url = (
                            f"{ORANGE_PUBLIC_BASE_URL}/{quote_plus(offer_id)}"
                        )
                        description = _clean_html(
                            item.get("description")
                            or item.get("profil")
                            or title
                        )

                        yield RawJobListing(
                            platform=self.name,
                            company=self.company_name,
                            title=title,
                            url=public_url,
                            description=description,
                            location=raw_loc,
                            raw_payload=item,
                        )

                except Exception as err:
                    print(
                        f"⚠️ [{self.name}] Error querying keyword "
                        f"'{keyword}': {err}"
                    )
                    continue

    def _is_location_allowed(self, location_str: str) -> bool:
        """Verify if job location matches target PACA/remote criteria.

        Args:
            location_str: Raw location string.

        Returns:
            bool: True if allowed or if whitelist empty, False otherwise.
        """
        if not self.allowed_locations:
            return True
        loc_lower = location_str.lower()
        return any(target in loc_lower for target in self.allowed_locations)

    @staticmethod
    def _extract_offers_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize JSON payload into a list of offer dictionaries.

        Args:
            payload: Deserialized JSON response from Orange jobs API.

        Returns:
            list[dict[str, Any]]: List of offer mappings.
        """
        if "offers" in payload and isinstance(payload["offers"], list):
            return payload["offers"]
        if "items" in payload and isinstance(payload["items"], list):
            return payload["items"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        return []

    @staticmethod
    def _extract_location(item: dict[str, Any]) -> str:
        """Extract a readable location string from offer data.

        Args:
            item: Single offer dictionary.

        Returns:
            str: Normalized location representation.
        """
        loc = item.get("location")
        if isinstance(loc, str):
            return loc
        if isinstance(loc, dict):
            city = loc.get("city", "")
            dpt = loc.get("department", "")
            return f"{city} ({dpt})".strip() if dpt else city
        city_direct = item.get("city", "")
        return str(city_direct) if city_direct else ""
