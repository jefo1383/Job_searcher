"""Scraper targeting VINCI corporate careers portal (VINCI Energies, Axians)."""

from collections.abc import AsyncIterator
import html
import json
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from core.location_matcher import LocationMatcher
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# En-têtes HTTP simulant un navigateur standard
DEFAULT_VINCI_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
}

# Racine du portail d'offres du groupe VINCI
VINCI_BASE_URL: str = "https://jobs.vinci.com"
VINCI_SEARCH_URL: str = "https://jobs.vinci.com/fr/recherche-d'offres"
VINCI_LOCATIONS_API: str = "https://jobs.vinci.com/fr/search-jobs/locations"
# Facette par défaut : Bouches-du-Rhône (alt=4) et PACA (alt=3) (réglable)
DEFAULT_BDR_FACET_ID: str = "3017382-2985244-3031359"
DEFAULT_BDR_FACET_TYPE: str = "4"
DEFAULT_PACA_FACET_ID: str = "3017382-2985244"
DEFAULT_PACA_FACET_TYPE: str = "3"


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
        facet_id: Radancy geographic hierarchy facet identifier.
        facet_type: Radancy geographic facet depth level.
    """

    def __init__(
        self,
        company_name: str = "Vinci",
        allowed_locations: list[str] | None = None,
        timeout_seconds: float = 15.0,
        facet_id: str = DEFAULT_BDR_FACET_ID,
        facet_type: str = DEFAULT_BDR_FACET_TYPE,
        location_matcher: LocationMatcher | None = None,
    ) -> None:
        """Initialize the Vinci scraper.

        Args:
            company_name: Name of hiring company.
            allowed_locations: List of target location tokens.
            timeout_seconds: Max network wait duration.
            facet_id: Default Radancy facet path (Bouches-du-Rhône).
            facet_type: Default Radancy facet level ("4"=dept, "3"=region).
            location_matcher: Shared geo validator with 25km radius check.
        """
        super().__init__(name="vinci")
        self.company_name = company_name
        self.allowed_locations = [
            loc.lower()
            for loc in (
                allowed_locations
                or [
                    "marseille",
                    "aix",
                    "vitrolles",
                    "bouches",
                    "13",
                    "remote",
                    "télétravail",
                ]
            )
        ]
        self.timeout_seconds = timeout_seconds
        self.facet_id = facet_id
        self.facet_type = facet_type
        self.location_matcher = location_matcher or LocationMatcher()

    async def _resolve_location_facet(
        self,
        client: httpx.AsyncClient,
        locations: list[str],
    ) -> tuple[str, str]:
        """Resolve a Radancy location facet ID/type from search criteria.

        Args:
            client: Reusable async HTTP client.
            locations: Requested target locations.

        Returns:
            tuple[str, str]: Pair of (facet_id, facet_type).
        """
        bdr_tokens = (
            "13",
            "bouches-du-rhône",
            "bouches-du-rhone",
            "marseille",
            "aix",
            "vitrolles",
        )
        lowered = [loc.strip().lower() for loc in locations if loc.strip()]
        if not lowered or any(tok in loc for loc in lowered for tok in bdr_tokens):
            return self.facet_id, self.facet_type

        if any("paca" in loc or "provence" in loc for loc in lowered):
            return DEFAULT_PACA_FACET_ID, DEFAULT_PACA_FACET_TYPE

        for loc in locations:
            cleaned = loc.strip()
            if not cleaned or cleaned.isdigit():
                continue
            try:
                resp = await client.get(
                    VINCI_LOCATIONS_API, params={"term": cleaned}
                )
                if resp.status_code == 200:
                    entries = resp.json()
                    if isinstance(entries, list) and entries:
                        first = entries[0]
                        lp = str(first.get("lp", "")).strip()
                        lt = str(first.get("lt", "")).strip()
                        if lp and lt:
                            return lp, lt
            except Exception:
                continue
        return self.facet_id, self.facet_type

    @staticmethod
    def _extract_job_posting_json_ld(html_content: str) -> dict[str, Any]:
        """Extract Schema.org JobPosting JSON-LD dictionary from detail HTML.

        Args:
            html_content: Raw HTML text of a VINCI offer page.

        Returns:
            dict[str, Any]: Parsed JobPosting payload or empty dict.
        """
        pattern = re.compile(
            r'<script[^>]*type=["\']application/ld'
            r'(?:\+|&#x2[Bb];|&#43;)json["\'][^>]*>(.*?)</script>',
            re.DOTALL | re.IGNORECASE,
        )
        for match in pattern.finditer(html_content):
            body = match.group(1).strip()
            if not body:
                continue
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("@type") == "JobPosting":
                return data
            if isinstance(data, list):
                for item in data:
                    if (
                        isinstance(item, dict)
                        and item.get("@type") == "JobPosting"
                    ):
                        return item
        return {}

    async def _fetch_full_description(
        self,
        client: httpx.AsyncClient,
        full_url: str,
        fallback_title: str,
        fallback_location: str,
    ) -> RawJobListing:
        """Fetch full offer description and metadata from a VINCI detail page.

        Args:
            client: Reusable async HTTP client.
            full_url: Absolute URL to the VINCI job detail page.
            fallback_title: Title parsed from the search results card.
            fallback_location: Location/department parsed from search card.

        Returns:
            RawJobListing: Enriched listing with full text and metadata.
        """
        company = self.company_name
        title = fallback_title
        location = fallback_location
        contract_type = ""
        body_text = ""
        payload: dict[str, Any] = {
            "scraped_url": full_url,
            "is_easy_apply": False,
        }

        try:
            response = await client.get(full_url)
            if response.status_code != 200:
                return RawJobListing(
                    platform=self.name,
                    company=company,
                    title=title,
                    url=full_url,
                    description=f"Poste VINCI / Axians : {title}",
                    location=location,
                    raw_payload=payload,
                )
        except httpx.HTTPError:
            return RawJobListing(
                platform=self.name,
                company=company,
                title=title,
                url=full_url,
                description=f"Poste VINCI / Axians : {title}",
                location=location,
                raw_payload=payload,
            )

        jp = self._extract_job_posting_json_ld(response.text)
        if jp:
            title = _clean_text(str(jp.get("title") or "")) or title
            hiring_org = jp.get("hiringOrganization")
            if isinstance(hiring_org, dict) and hiring_org.get("name"):
                company = _clean_text(str(hiring_org["name"])) or company
            contract_type = _clean_text(str(jp.get("employmentType") or ""))

            job_loc = jp.get("jobLocation")
            loc_entry = (
                job_loc[0]
                if isinstance(job_loc, list) and job_loc
                else job_loc
            )
            if isinstance(loc_entry, dict):
                addr = loc_entry.get("address")
                if isinstance(addr, dict):
                    locality = _clean_text(
                        str(
                            addr.get("addressLocality")
                            or addr.get("streetAddress")
                            or ""
                        )
                    )
                    if locality:
                        location = locality

            body_text = _clean_text(str(jp.get("description") or ""))
            edu = _clean_text(str(jp.get("educationRequirements") or ""))
            if edu and edu not in body_text:
                body_text = f"{body_text}\nFormation : {edu}".strip()

        # Repli sur le bloc HTML .ats-description si le JSON-LD est absent/vide
        if not body_text:
            ats_match = re.search(
                r'<div[^>]*class=["\'][^"\']*ats-description[^"\']*["\'][^>]*>'
                r'(.*?)</div>\s*(?:<div|</section|<section)',
                response.text,
                re.DOTALL | re.IGNORECASE,
            )
            if ats_match:
                body_text = _clean_text(ats_match.group(1))

        if contract_type:
            payload["contract_type"] = contract_type

        header_lines = [f"{title} chez {company}"]
        meta_line = f"Lieu : {location}"
        if contract_type:
            meta_line += f" | Contrat : {contract_type}"
        header_lines.append(meta_line)

        full_description = (
            "\n".join(header_lines) + f"\n\nDescription :\n{body_text}"
            if body_text
            else f"Poste VINCI / Axians : {title} ({location})"
        )

        return RawJobListing(
            platform=self.name,
            company=company,
            title=title,
            url=full_url,
            description=full_description,
            location=location,
            raw_payload=payload,
        )

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
        seen_urls: set[str] = set()

        async with httpx.AsyncClient(
            headers=DEFAULT_VINCI_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            facet_id, facet_type = await self._resolve_location_facet(
                client=client,
                locations=locations,
            )

            # Motifs des cartes Radancy (/fr/emploi/<ville>/<slug>/<org>/<id>)
            card_pattern = re.compile(
                r'<a([^>]+)href=["\'](/fr/emploi/[^"\']+)["\']([^>]*)>(.*?)</a>',
                re.DOTALL | re.IGNORECASE,
            )
            title_span_pattern = re.compile(
                r'<span[^>]*search-results--link-jobtitle[^>]*>(.*?)</span>',
                re.DOTALL | re.IGNORECASE,
            )
            industry_pattern = re.compile(
                r'data-industry=["\']([^"\']*)["\']',
                re.IGNORECASE,
            )

            for keyword in keywords:
                if found_count >= limit:
                    break

                # Paramètre ?k=... requis par Radancy pour filtrer par mot-clé
                query_url = (
                    f"{VINCI_SEARCH_URL}?"
                    f"k={quote_plus(keyword)}&"
                    f"alp={quote_plus(facet_id)}&"
                    f"alt={quote_plus(facet_type)}"
                )

                try:
                    response = await client.get(query_url)
                    if response.status_code != 200:
                        continue

                    for match in card_pattern.finditer(response.text):
                        if found_count >= limit:
                            break

                        attrs_before, rel_url, attrs_after, inner_html = (
                            match.groups()
                        )
                        if "/recherche" in rel_url.lower():
                            continue

                        full_url = f"{VINCI_BASE_URL}{rel_url}"
                        if full_url in seen_urls:
                            continue
                        seen_urls.add(full_url)

                        span_match = title_span_pattern.search(inner_html)
                        title = _clean_text(
                            span_match.group(1) if span_match else inner_html
                        )
                        if not title or is_excluded_title(title):
                            continue

                        # Extraction du département et de la ville depuis la carte
                        ind_match = industry_pattern.search(
                            f"{attrs_before} {attrs_after}"
                        )
                        dept = (
                            _clean_text(ind_match.group(1))
                            if ind_match
                            else ""
                        )
                        url_parts = rel_url.strip("/").split("/")
                        city_slug = (
                            url_parts[2].replace("-", " ").title()
                            if len(url_parts) >= 3
                            else ""
                        )
                        fallback_loc = (
                            f"{city_slug} ({dept})".strip()
                            if city_slug and dept
                            else city_slug or dept or "Bouches-du-Rhône (13)"
                        )
                        if city_slug and not (
                            await self.location_matcher.is_location_allowed(
                                raw_location=city_slug,
                                client=client,
                            )
                        ):
                            continue

                        enriched = await self._fetch_full_description(
                            client=client,
                            full_url=full_url,
                            fallback_title=title,
                            fallback_location=fallback_loc,
                        )
                        if is_excluded_title(enriched.title):
                            continue
                        if not await self.location_matcher.is_location_allowed(
                            raw_location=enriched.location,
                            client=client,
                        ):
                            continue

                        yield enriched
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
