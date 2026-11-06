"""Scraper implementation for Welcome to the Jungle platform."""

from collections.abc import AsyncIterator
import re
from typing import Any
import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# Constantes pour l'API publique de recherche WTTJ
WTTJ_ALGOLIA_URL: str = (
    "https://csekhvms53-dsn.algolia.net/1/indexes/wttj_jobs_fr/query"
)
WTTJ_APP_ID: str = "CSEKHVMS53"
WTTJ_API_KEY: str = "48b4792a71e22707297e644cb3db67db"


def _clean_html_text(raw_text: str | None) -> str:
    """Remove HTML tags and collapse whitespace.

    Args:
        raw_text: Raw string possibly containing HTML markup.

    Returns:
        str: Cleaned plain text string.
    """
    if not raw_text:
        return ""
    stripped = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", stripped).strip()


class WTTJScraper(BaseScraper):
    """Scraper targeting Welcome to the Jungle using public search index."""

    def __init__(self) -> None:
        """Initialize the Welcome to the Jungle scraper."""
        super().__init__(name="wttj")

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch listings from Welcome to the Jungle matching keywords.

        Args:
            keywords: Target keywords to query.
            locations: Geographical filters (e.g. 'Marseille', 'Remote').
            limit: Maximum count of listings to return.

        Yields:
            RawJobListing: Normalized raw job post items.
        """
        headers = {
            "x-algolia-application-id": WTTJ_APP_ID,
            "x-algolia-api-key": WTTJ_API_KEY,
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
        }

        yielded_count = 0
        async with httpx.AsyncClient(timeout=15.0) as client:
            for keyword in keywords:
                if yielded_count >= limit:
                    break

                payload: dict[str, Any] = {
                    "query": keyword,
                    "hitsPerPage": min(limit - yielded_count, 30),
                }

                try:
                    # Requête asynchrone à l'index de recherche WTTJ
                    response = await client.post(
                        WTTJ_ALGOLIA_URL,
                        headers=headers,
                        json=payload,
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    hits = data.get("hits", [])

                    for hit in hits:
                        if yielded_count >= limit:
                            break

                        org = hit.get("organization", {})
                        org_name = org.get("name", "Entreprise Inconnue")
                        org_slug = org.get("slug", "")
                        job_title = hit.get("name", "")
                        job_slug = hit.get("slug", "")

                        if not org_slug or not job_slug:
                            continue

                        # Construction de l'URL pérenne de l'offre
                        job_url = (
                            f"https://www.welcometothejungle.com/fr/"
                            f"companies/{org_slug}/jobs/{job_slug}"
                        )

                        office = hit.get("office", {})
                        city = office.get("city", "") if office else ""
                        contract = hit.get("contract_type", "")

                        # Filtrage géographique sommaire si spécifié
                        if locations and city:
                            city_lower = city.lower()
                            if not any(
                                loc.lower() in city_lower for loc in locations
                            ):
                                # Vérifie si l'offre propose le remote
                                is_remote = (
                                    hit.get("remote") in ("full", "partial")
                                )
                                if not is_remote:
                                    continue

                        desc_clean = _clean_html_text(
                            hit.get("description", "")
                        )
                        profile_clean = _clean_html_text(hit.get("profile", ""))
                        full_desc = (
                            f"{job_title} chez {org_name}\n"
                            f"Lieu : {city or 'Non spécifié'} | "
                            f"Contrat : {contract}\n\n"
                            f"Description :\n{desc_clean}\n\n"
                            f"Profil recherché :\n{profile_clean}"
                        )

                        yield RawJobListing(
                            platform=self.name,
                            company=org_name,
                            title=job_title,
                            url=job_url,
                            description=full_desc,
                            location=city,
                            raw_payload=hit,
                        )
                        yielded_count += 1

                except httpx.HTTPError as err:
                    # Gestion gracieuse des erreurs réseau ou indisponibilités
                    print(
                        f"⚠️ [WTTJScraper] Network warning for query "
                        f"'{keyword}': {err}"
                    )
                    continue
