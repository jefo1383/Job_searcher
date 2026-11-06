"""Scraper targeting Groupe BPCE career portal (Banque Populaire, Caisse d'Epargne)."""

import html
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# En-têtes HTTP simulant un navigateur standard
DEFAULT_BPCE_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Points d'accès de recherche du groupe BPCE
BPCE_BASE_URL: str = "https://recrutement.bpce.fr"
BPCE_API_SEARCH_URL: str = (
    "https://recrutement.bpce.fr/app/wp-json/bpce/v1/search/jobs"
)


def _clean_text(raw_text: str | None) -> str:
    """Sanitize text by stripping HTML markup and extra whitespaces.

    Args:
        raw_text: Raw string with potential HTML tags.

    Returns:
        str: Cleaned plain text.
    """
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    clean_text = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", clean_text).strip()


class BpceScraper(BaseScraper):
    """Scraper extracting positions from Groupe BPCE career portal.

    Attributes:
        company_name: Hiring organization identifier.
        allowed_locations: Whitelist of target locations (Bouches-du-Rhône).
        timeout_seconds: Request timeout duration in seconds.
    """

    def __init__(
        self,
        company_name: str = "Groupe BPCE",
        allowed_locations: list[str] | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        """Initialize the BPCE scraper.

        Args:
            company_name: Name of hiring company.
            allowed_locations: List of target location tokens.
            timeout_seconds: Max network wait duration.
        """
        super().__init__(name="bpce")
        self.company_name = company_name
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
        """Query BPCE recruitment portal via official WordPress search API.

        Args:
            keywords: Target role or technology keywords.
            locations: Secondary geographic search criteria.
            limit: Maximum count of listings to extract per search.

        Yields:
            RawJobListing: Normalized scraped job post.
        """
        yielded_count = 0
        async with httpx.AsyncClient(
            headers=DEFAULT_BPCE_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for keyword in keywords:
                if yielded_count >= limit:
                    break

                payload: dict[str, Any] = {
                    "lang": "fr",
                    "keyword": keyword,
                    "size": str(min(limit - yielded_count, 20)),
                    "from": "0",
                }

                try:
                    response = await client.post(
                        BPCE_API_SEARCH_URL,
                        json=payload,
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    items = data.get("data", {}).get("items", [])

                    for item in items:
                        if yielded_count >= limit:
                            break

                        title = _clean_text(item.get("title", ""))
                        if not title:
                            continue

                        # Extraction de la marque (ex: CEPAC, BPCE IT, Natixis)
                        brands = item.get("brand", [])
                        brand_name = (
                            brands[0]
                            if (isinstance(brands, list) and brands)
                            else self.company_name
                        )

                        # Extraction de la localisation
                        raw_loc = item.get("localisation", "")
                        if not raw_loc:
                            loc_list = item.get("localisations", [])
                            if isinstance(loc_list, list) and loc_list:
                                first_loc = loc_list[0]
                                raw_loc = (
                                    first_loc.get("city")
                                    or first_loc.get("localisation")
                                    or first_loc.get("department")
                                    or ""
                                )

                        # Filtrage géographique
                        if not self._is_location_allowed(raw_loc):
                            continue

                        # Extraction des URLs de consultation et de candidature directes
                        link_dict = item.get("link", {})
                        rel_url = (
                            link_dict.get("url", "")
                            if isinstance(link_dict, dict)
                            else str(link_dict)
                        )
                        public_url = urljoin(BPCE_BASE_URL, rel_url)

                        postulate_dict = item.get("postulate_link", {})
                        apply_url = (
                            postulate_dict.get("url")
                            if isinstance(postulate_dict, dict)
                            else None
                        )

                        desc = _clean_text(item.get("description", "")) or title

                        yield RawJobListing(
                            platform=self.name,
                            company=brand_name,
                            title=title,
                            url=public_url,
                            description=desc,
                            location=raw_loc,
                            raw_payload={
                                "brand": brand_name,
                                "apply_url": apply_url or public_url,
                                "raw_item": item,
                            },
                        )
                        yielded_count += 1

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
