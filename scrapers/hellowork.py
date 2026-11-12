"""Scraper implementation for HelloWork using HTTP async and JSON-LD parsing."""

from collections.abc import AsyncIterator
import html
import json
import re
from typing import Any
from urllib.parse import urljoin
import httpx

from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# URL racine de recherche d'offres HelloWork
HELLOWORK_SEARCH_URL: str = (
    "https://hellowork.com/fr-fr/emploi/recherche.html"
)
BASE_DOMAIN: str = "https://hellowork.com"
# Seuil minimal (caractères) avant chargement de la fiche complète (réglable)
MIN_FULL_DESCRIPTION_CHARS: int = 400
# Expression régulière de détection du type de contrat en français
FRENCH_CONTRACT_PATTERN: re.Pattern[str] = re.compile(
    r"\b(Alternance|Contrat\s+pro(?:fessionnalisation)?|"
    r"Apprentissage|Stage|CDI|CDD|Intérim|Interim|Freelance)\b",
    re.IGNORECASE,
)


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

    def __init__(
        self,
        default_location: str = "Bouches-du-Rhône",
        min_full_description_chars: int = MIN_FULL_DESCRIPTION_CHARS,
    ) -> None:
        """Initialize the HelloWork scraper.

        Args:
            default_location: Primary geographical filter.
            min_full_description_chars: Threshold below which the detail
                page is fetched to retrieve the complete offer text.
        """
        super().__init__(name="hellowork")
        self.default_location = default_location
        self.min_full_description_chars = min_full_description_chars

    @staticmethod
    def _extract_french_contract(html_content: str, fallback: str = "") -> str:
        """Detect a French contract label from page title or visible markup.

        Args:
            html_content: Raw HTML text of the job page or card.
            fallback: Fallback contract value (e.g. from search card).

        Returns:
            str: Detected French contract string or fallback.
        """
        if fallback:
            return fallback
        title_match = re.search(
            r"<title>(.*?)</title>", html_content, re.DOTALL | re.IGNORECASE
        )
        if title_match:
            clean_title = _clean_html_text(title_match.group(1))
            contract_match = FRENCH_CONTRACT_PATTERN.search(clean_title)
            if contract_match:
                return contract_match.group(1).strip()
        return ""

    def _extract_locality(self, job_location: Any) -> str:
        """Extract city/locality from Schema.org jobLocation payload.

        Args:
            job_location: Dict or list of dicts representing jobLocation.

        Returns:
            str: Extracted locality or default scraper location.
        """
        loc_entry = (
            job_location[0]
            if isinstance(job_location, list) and job_location
            else job_location
        )
        if not isinstance(loc_entry, dict):
            return self.default_location
        address_dict = loc_entry.get("address", {})
        if not isinstance(address_dict, dict):
            return self.default_location
        locality = address_dict.get("addressLocality") or address_dict.get(
            "addressRegion"
        )
        return str(locality).strip() if locality else self.default_location

    def _extract_from_json_ld(self, html_content: str) -> list[RawJobListing]:
        """Extract structured Schema.org JobPosting objects from script tags.

        Args:
            html_content: Raw HTML page text.

        Returns:
            list[RawJobListing]: List of successfully parsed listings.
        """
        listings: list[RawJobListing] = []
        # Recherche de toutes les balises application/ld+json (avec + encodé ou brut)
        script_pattern = re.compile(
            r'<script[^>]*type=[\'"]application/ld'
            r'(?:\+|&#x2[Bb];|&#43;)json[\'"][^>]*>(.*?)</script>',
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
                title = _clean_html_text(str(item.get("title", "")))
                raw_url = str(item.get("url", ""))
                if not title:
                    continue

                full_url = (
                    raw_url
                    if raw_url.startswith("http")
                    else urljoin(BASE_DOMAIN, raw_url)
                    if raw_url
                    else ""
                )

                hiring_org = item.get("hiringOrganization", {})
                company_name = (
                    str(hiring_org.get("name", "Entreprise confidentielle"))
                    if isinstance(hiring_org, dict)
                    else "Entreprise confidentielle"
                )

                locality = self._extract_locality(item.get("jobLocation"))
                description = _clean_html_text(
                    str(item.get("description", ""))
                )
                extra_sections: list[str] = []
                for field_key, label in (
                    ("skills", "Compétences"),
                    ("experienceRequirements", "Expérience requise"),
                    ("educationRequirements", "Formation"),
                ):
                    raw_val = item.get(field_key)
                    if isinstance(raw_val, list):
                        joined = ", ".join(
                            _clean_html_text(str(v)) for v in raw_val if v
                        )
                        if joined:
                            extra_sections.append(f"{label} : {joined}")
                    elif isinstance(raw_val, dict):
                        val_text = _clean_html_text(
                            str(raw_val.get("monthsOfExperience") or raw_val)
                        )
                        if val_text:
                            extra_sections.append(f"{label} : {val_text}")
                    elif raw_val:
                        cleaned_val = _clean_html_text(str(raw_val))
                        if cleaned_val:
                            extra_sections.append(f"{label} : {cleaned_val}")

                french_contract = self._extract_french_contract(html_content)
                schema_contract = str(item.get("employmentType", "")).strip()
                contract_type = french_contract or schema_contract

                body_parts = [description] if description else []
                body_parts.extend(extra_sections)
                combined_body = "\n".join(body_parts)

                full_desc = (
                    f"{title} chez {company_name}\n"
                    f"Lieu : {locality} | Contrat : {contract_type}\n\n"
                    f"Description :\n{combined_body}"
                )

                raw_item_payload = dict(item)
                raw_item_payload["is_easy_apply"] = False
                if contract_type:
                    raw_item_payload["contract_type"] = contract_type

                listings.append(
                    RawJobListing(
                        platform=self.name,
                        company=company_name,
                        title=title,
                        url=full_url,
                        description=full_desc,
                        location=locality,
                        raw_payload=raw_item_payload,
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

            # Détection d'un nom d'entreprise en suffixe (ex: 'Ingénieur IA H/F Klanik')
            company_name = "Entreprise"
            comp_match = re.search(
                r"(?:H/F|F/H|\bchez\b|[-–—])\s+([A-Za-z0-9_\s&]{2,35})$",
                title_clean,
                re.IGNORECASE,
            )
            if comp_match:
                company_name = comp_match.group(1).strip()

            # Recherche du badge de contrat dans la carte autour du lien
            card_slice = html_content[match.end() : match.end() + 700]
            next_card_idx = card_slice.find("/fr-fr/emplois/")
            if next_card_idx != -1:
                card_slice = card_slice[:next_card_idx]
            card_text = _clean_html_text(card_slice)
            contract_match = FRENCH_CONTRACT_PATTERN.search(
                f"{title_clean} {card_text}"
            )
            detected_contract = (
                contract_match.group(1).strip() if contract_match else ""
            )

            desc_text = (
                f"{title_clean} - Contrat : {detected_contract} "
                "- Offre consultable sur HelloWork"
                if detected_contract
                else f"{title_clean} - Offre consultable sur HelloWork"
            )
            payload: dict[str, Any] = {"path": path, "is_easy_apply": False}
            if detected_contract:
                payload["contract_type"] = detected_contract

            listings.append(
                RawJobListing(
                    platform=self.name,
                    company=company_name,
                    title=title_clean,
                    url=full_url,
                    description=desc_text,
                    location=self.default_location,
                    raw_payload=payload,
                )
            )

        return listings

    async def _fetch_full_description(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        listing: RawJobListing,
    ) -> RawJobListing:
        """Fetch full job details from a HelloWork posting page.

        Args:
            client: Reusable async HTTP client.
            headers: Browser-like HTTP request headers.
            listing: Initial summary listing from search cards.

        Returns:
            RawJobListing: Enriched listing with full text when available.
        """
        try:
            response = await client.get(listing.url, headers=headers)
            if response.status_code != 200:
                return listing
        except httpx.HTTPError:
            return listing

        card_contract = str(
            (listing.raw_payload or {}).get("contract_type", "")
        ).strip()

        # 1. Extraction prioritaire du JobPosting JSON-LD sur la fiche détaillée
        detail_jobs = self._extract_from_json_ld(response.text)
        if detail_jobs:
            detail = detail_jobs[0]
            detail_payload = dict(detail.raw_payload or {})
            detail_contract = str(
                detail_payload.get("contract_type", "")
            ).strip()
            resolved_contract = self._extract_french_contract(
                response.text,
                fallback=card_contract,
            ) or detail_contract
            if resolved_contract:
                detail_payload["contract_type"] = resolved_contract
            detail_payload["is_easy_apply"] = False

            resolved_company = (
                detail.company
                if detail.company and detail.company != "Entreprise confidentielle"
                else listing.company
            )
            return RawJobListing(
                platform=self.name,
                company=resolved_company,
                title=detail.title or listing.title,
                url=listing.url,
                description=detail.description or listing.description,
                location=detail.location or listing.location,
                raw_payload=detail_payload,
            )

        # 2. Repli sur le contenu principal HTML si aucun JSON-LD présent
        main_match = re.search(
            r"<main[^>]*>(.*?)</main>",
            response.text,
            re.DOTALL | re.IGNORECASE,
        )
        raw_body = main_match.group(1) if main_match else response.text
        cleaned_body = _clean_html_text(raw_body)
        if len(cleaned_body) > len(listing.description):
            merged_payload = dict(listing.raw_payload or {})
            merged_payload["is_easy_apply"] = False
            resolved_contract = self._extract_french_contract(
                response.text, fallback=card_contract
            )
            if resolved_contract:
                merged_payload["contract_type"] = resolved_contract
            return RawJobListing(
                platform=self.name,
                company=listing.company,
                title=listing.title,
                url=listing.url,
                description=cleaned_body[:8000],
                location=listing.location,
                raw_payload=merged_payload,
            )

        return listing

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
        seen_urls: set[str] = set()

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
                        if job.url in seen_urls:
                            continue
                        seen_urls.add(job.url)

                        # Pré-filtre par titre avant de charger la fiche complète
                        if is_excluded_title(job.title):
                            continue

                        enriched = job
                        if len(job.description) < self.min_full_description_chars:
                            enriched = await self._fetch_full_description(
                                client=client,
                                headers=headers,
                                listing=job,
                            )
                            if is_excluded_title(enriched.title):
                                continue

                        yield enriched
                        yielded_count += 1

                except httpx.HTTPError as err:
                    print(
                        f"⚠️ [HelloWorkScraper] Network warning for query "
                        f"'{keyword}': {err}"
                    )
                    continue
