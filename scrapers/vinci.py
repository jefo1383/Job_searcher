"""Scraper targeting VINCI corporate careers portal (VINCI Energies, Axians)."""

import html
import re
from collections.abc import AsyncIterator
from urllib.parse import quote_plus

import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# En-têtes HTTP simulant un navigateur standard
DEFAULT_VINCI_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/html, */*",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Racine du portail d'offres du groupe VINCI
VINCI_BASE_URL: str = "https://jobs.vinci.com"
VINCI_SEARCH_URL: str = "https://jobs.vinci.com/fr/recherche-d'offres"


def _clean_text(raw_text: str | None) -> str:
    """Sanitize raw text by stripping HTML tags and condensing whitespace.

    Args:
        raw_text: Raw string containing possible HTML entities.

    Returns:
        str: Cleaned plain text.
    """
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    clean_text = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", clean_text).strip()


class VinciScraper(BaseScraper):
    """Scraper extracting positions from VINCI and Axians careers portal.

    Attributes:
        company_name: Hiring organization identifier.
        allowed_locations: Whitelist of target locations (Aix, Marseille, Vitrolles).
        timeout_seconds: Request timeout duration in seconds.
    """

    def __init__(
        self,
        company_name: str = "Vinci",
        allowed_locations: list[str] | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        """Initialize the Vinci scraper.

        Args:
            company_name: Name of hiring company.
            allowed_locations: List of target location tokens.
            timeout_seconds: Max network wait duration.
        """
        super().__init__(name="vinci")
        self.company_name = company_name
        self.allowed_locations = [
            loc.lower()
            for loc in (
                allowed_locations
                or ["marseille", "aix", "vitrolles", "13", "remote", "télétravail"]
            )
        ]
        self.timeout_seconds = timeout_seconds

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Query VINCI jobs portal for matching positions.

        Args:
            keywords: Target role or technology keywords.
            locations: Secondary geographic search criteria.
            limit: Maximum count of listings to extract per search.

        Yields:
            RawJobListing: Normalized scraped job post.
        """
        found_count = 0
        async with httpx.AsyncClient(
            headers=DEFAULT_VINCI_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for keyword in keywords:
                if found_count >= limit:
                    break

                # Requête de recherche sur le portail VINCI
                query_url = (
                    f"{VINCI_SEARCH_URL}/{quote_plus(keyword)}/13"
                )

                try:
                    response = await client.get(query_url)
                    if response.status_code != 200:
                        continue

                    # 1. Extraction d'offres depuis les cartes HTML structurées
                    card_pattern = re.compile(
                        r'<a[^>]+href=["\'](/job/[^"\']+)["\'][^>]*>(.*?)</a>',
                        re.DOTALL | re.IGNORECASE,
                    )

                    for match in card_pattern.finditer(response.text):
                        if found_count >= limit:
                            break

                        rel_url, raw_title = match.groups()
                        title = _clean_text(raw_title)
                        if not title:
                            continue

                        full_url = f"{VINCI_BASE_URL}{rel_url}"

                        # Vérification de l'URL pour éviter les liens d'onglets
                        if "/recherche" in rel_url.lower():
                            continue

                        yield RawJobListing(
                            platform=self.name,
                            company=self.company_name,
                            title=title,
                            url=full_url,
                            description=f"Poste VINCI / Axians : {title}",
                            location="Bouches-du-Rhône (13)",
                            raw_payload={"scraped_url": full_url},
                        )
                        found_count += 1

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
