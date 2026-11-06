"""Scraper implementation for HelloWork using HTTP async and JSON-LD parsing."""

from collections.abc import AsyncIterator
import html
import json
import re
from typing import Any
from urllib.parse import urljoin
import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# URL racine de recherche d'offres HelloWork
HELLOWORK_SEARCH_URL: str = (
    "https://www.hellowork.com/fr-fr/emploi/recherche.html"
)
BASE_DOMAIN: str = "https://www.hellowork.com"


def _clean_html_text(raw_text: str | None) -> str:
    """Strip HTML markup, unescape entities, and condense whitespaces.

    Args:
        raw_text: Raw string containing potential HTML markup.

    Returns:
        str: Sanitized plain text string.
    """
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


class HelloWorkScraper(BaseScraper):
    """Asynchronous scraper extracting job postings from HelloWork.

    Attributes:
        default_location: Default department or city targeted for search.
    """

    def __init__(self, default_location: str = "Bouches-du-Rhône") -> None:
        """Initialize the HelloWork scraper.

        Args:
            default_location: Primary geographical filter.
        """
        super().__init__(name="hellowork")
        self.default_location = default_location

    def _extract_from_json_ld(self, html_content: str) -> list[RawJobListing]:
        """Extract structured Schema.org JobPosting objects from script tags.

        Args:
            html_content: Raw HTML page text.

        Returns:
            list[RawJobListing]: List of successfully parsed listings.
        """
        listings: list[RawJobListing] = []
        # Recherche de toutes les balises application/ld+json
        script_pattern = re.compile(
            r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in script_pattern.finditer(html_content):
            script_body = match.group(1).strip()
            if not script_body:
                continue

            try:
                data = json.loads(script_body)
            except json.JSONDecodeError:
                continue

            items_to_check: list[dict[str, Any]] = []
            if isinstance(data, dict):
                if data.get("@type") == "JobPosting":
                    items_to_check.append(data)
                elif data.get("@type") == "ItemList":
                    elements = data.get("itemListElement", [])
                    for elem in elements:
                        if isinstance(elem, dict):
                            item = elem.get("item", elem)
                            if (
                                isinstance(item, dict)
                                and item.get("@type") == "JobPosting"
                            ):
                                items_to_check.append(item)
            elif isinstance(data, list):
                for elem in data:
                    if (
                        isinstance(elem, dict)
                        and elem.get("@type") == "JobPosting"
                    ):
                        items_to_check.append(elem)

            for item in items_to_check:
                title = item.get("title", "")
                raw_url = item.get("url", "")
                if not title or not raw_url:
                    continue

                full_url = (
                    raw_url
                    if raw_url.startswith("http")
                    else urljoin(BASE_DOMAIN, raw_url)
                )

                hiring_org = item.get("hiringOrganization", {})
                company_name = (
                    hiring_org.get("name", "Entreprise confidentielle")
                    if isinstance(hiring_org, dict)
                    else "Entreprise confidentielle"
                )

                location_dict = item.get("jobLocation", {})
                address_dict = (
                    location_dict.get("address", {})
                    if isinstance(location_dict, dict)
                    else {}
                )
                locality = (
                    address_dict.get("addressLocality", self.default_location)
                    if isinstance(address_dict, dict)
                    else self.default_location
                )

                description = _clean_html_text(item.get("description", ""))
                contract_type = item.get("employmentType", "")

                full_desc = (
                    f"{title} chez {company_name}\n"
                    f"Lieu : {locality} | Contrat : {contract_type}\n\n"
                    f"Description :\n{description}"
                )

                listings.append(
                    RawJobListing(
                        platform=self.name,
                        company=company_name,
                        title=title,
                        url=full_url,
                        description=full_desc,
                        location=locality,
                        raw_payload=item,
                    )
                )

        return listings

    def _extract_from_html_cards(
        self, html_content: str
    ) -> list[RawJobListing]:
        """Fallback extraction parsing HTML job cards using regex patterns.

        Args:
            html_content: Raw HTML text of the search results page.

        Returns:
            list[RawJobListing]: Extracted job listings from cards.
        """
        listings: list[RawJobListing] = []

        # Détection des liens d'annonces HelloWork (/fr-fr/emplois/<id>.html)
        link_pattern = re.compile(
            r'<a[^>]+href=[\'"](/fr-fr/emplois/[^\'"]+\.html)[\'"][^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        seen_urls: set[str] = set()

        for match in link_pattern.finditer(html_content):
            path = match.group(1).strip()
            anchor_html = match.group(2)
            full_url = urljoin(BASE_DOMAIN, path)

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Extraction du titre textuel du lien
            title_clean = _clean_html_text(anchor_html)
            if len(title_clean) < 3 or "voir l'offre" in title_clean.lower():
                continue

            listings.append(
                RawJobListing(
                    platform=self.name,
                    company="HelloWork Listing",
                    title=title_clean,
                    url=full_url,
                    description=f"{title_clean} - Offre consultable sur HelloWork",
                    location=self.default_location,
                    raw_payload={"path": path},
                )
            )

        return listings

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch job postings from HelloWork matching target queries.

        Args:
            keywords: Target keywords to query.
            locations: Geographical focus (falls back to default).
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Normalized raw job post items.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
        }

        search_loc = locations[0] if locations else self.default_location
        yielded_count = 0

        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True
        ) as client:
            for keyword in keywords:
                if yielded_count >= limit:
                    break

                params = {
                    "k": keyword,
                    "l": search_loc,
                    "ray": "20",
                }

                try:
                    # Requête asynchrone à la page de recherche HelloWork
                    response = await client.get(
                        HELLOWORK_SEARCH_URL,
                        params=params,
                        headers=headers,
                    )
                    if response.status_code != 200:
                        continue

                    # 1. Extraction prioritaire depuis les données structurées JSON-LD
                    found_jobs = self._extract_from_json_ld(response.text)

                    # 2. Repli sur les cartes HTML si aucun JSON-LD exploitable
                    if not found_jobs:
                        found_jobs = self._extract_from_html_cards(
                            response.text
                        )

                    for job in found_jobs:
                        if yielded_count >= limit:
                            break
                        yield job
                        yielded_count += 1

                except httpx.HTTPError as err:
                    print(
                        f"⚠️ [HelloWorkScraper] Network warning for query "
                        f"'{keyword}': {err}"
                    )
                    continue
