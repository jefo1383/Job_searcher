"""Scraper targeting Groupe BPCE career portal (Banque Populaire, Caisse d'Epargne)."""

import html
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote_plus

import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# En-têtes HTTP simulant un navigateur standard
DEFAULT_BPCE_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/html, */*",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Points d'accès de recherche du groupe BPCE
BPCE_BASE_URL: str = "https://recrutement.bpce.fr"
BPCE_API_SEARCH_URL: str = "https://recrutement.bpce.fr/api/jobs/search"


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
        """Query BPCE recruitment portal for matching jobs.

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

                # 1. Tentative via l'endpoint de recherche API
                params: dict[str, str | int] = {
                    "keyword": keyword,
                    "location": "13",
                    "limit": min(limit - yielded_count, 20),
                }

                try:
                    response = await client.get(
                        BPCE_API_SEARCH_URL,
                        params=params,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        jobs_list = self._extract_jobs(data)

                        for item in jobs_list:
                            if yielded_count >= limit:
                                break

                            title = item.get("title", "").strip()
                            loc = item.get("location", "")
                            job_url = item.get("url", "")
                            desc = _clean_text(
                                item.get("description")
                                or item.get("summary")
                                or title
                            )

                            if not title or not job_url:
                                continue

                            if not self._is_location_allowed(loc):
                                continue

                            if not job_url.startswith("http"):
                                job_url = f"{BPCE_BASE_URL}{job_url}"

                            yield RawJobListing(
                                platform=self.name,
                                company=self.company_name,
                                title=title,
                                url=job_url,
                                description=desc,
                                location=loc,
                                raw_payload=item,
                            )
                            yielded_count += 1

                    else:
                        # 2. Fallback via recherche HTML SSR
                        search_page_url = (
                            f"{BPCE_BASE_URL}/recherche-offres/"
                            f"{quote_plus(keyword)}/13"
                        )
                        page_res = await client.get(search_page_url)
                        if page_res.status_code != 200:
                            continue

                        # Extraction par carte HTML d'offres
                        card_pattern = re.compile(
                            r'<a[^>]+href=["\'](/offre-de-emploi/[^"\']+)["\']'
                            r"[^>]*>(.*?)</a>",
                            re.DOTALL | re.IGNORECASE,
                        )
                        for match in card_pattern.finditer(page_res.text):
                            if yielded_count >= limit:
                                break

                            relative_url, raw_title = match.groups()
                            clean_t = _clean_text(raw_title)
                            if not clean_t:
                                continue

                            full_url = f"{BPCE_BASE_URL}{relative_url}"
                            yield RawJobListing(
                                platform=self.name,
                                company=self.company_name,
                                title=clean_t,
                                url=full_url,
                                description=f"Offre BPCE : {clean_t}",
                                location="Marseille (13)",
                                raw_payload={"scraped_url": full_url},
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

    @staticmethod
    def _extract_jobs(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize JSON payload into a list of job mappings.

        Args:
            payload: JSON response from BPCE recruitment endpoint.

        Returns:
            list[dict[str, Any]]: List of job entries.
        """
        if "jobs" in payload and isinstance(payload["jobs"], list):
            return payload["jobs"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        return []
