"""Dedicated scraper targeting Thales career portal (Phenom People SSR)."""

from collections.abc import AsyncIterator
import html
import json
import re
from typing import Any
import unicodedata
from urllib.parse import quote_plus

import httpx

from core.location_matcher import LocationMatcher
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# URLs publiques du portail carrières Phenom de Thales
THALES_SEARCH_URL: str = (
    "https://careers.thalesgroup.com/fr/fr/search-results"
)
THALES_JOB_URL_TEMPLATE: str = (
    "https://careers.thalesgroup.com/fr/fr/job/{job_seq_no}"
)

# Paramètres réglables de pagination et de filtrage géographique/IT
DEFAULT_THALES_AREA_QUERIES: tuple[str, ...] = (
    "Aix-en-Provence",
    "La Ciotat",
    "Marseille",
    "Aubagne",
    "Bouches-du-Rhone",
)
DEFAULT_THALES_PAGE_SIZE: int = 10
DEFAULT_THALES_MAX_PAGES: int = 8
DEFAULT_THALES_ALLOWED_LOCATIONS: tuple[str, ...] = (
    "aubagne",
    "la ciotat",
    "ciotat",
    "aix",
    "marseille",
    "gémenos",
    "gemenos",
    "meyreuil",
    "marignane",
    "vitrolles",
    "13",
    "bouches-du-rhone",
    "remote",
    "télétravail",
)

# Signaux IT bilingues pour retenir les offres techniques malgré un titre anglais
DEFAULT_THALES_IT_TOKENS: tuple[str, ...] = (
    "ia",
    "ai",
    "intelligence artificielle",
    "data",
    "python",
    "software",
    "logiciel",
    "developp",
    "developer",
    "dev",
    "devops",
    "devsecops",
    "secops",
    "cloud",
    "finops",
    "sre",
    "cyber",
    "soc",
    "csirt",
    "system",
    "systeme",
    "reseau",
    "network",
    "iot",
    "embarque",
    "embedded",
    "digital",
    "numerique",
    "informatique",
    "algorith",
    "r&d",
    "automation",
    "automatisation",
)

DEFAULT_THALES_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


def _strip_accents_lower(text: str) -> str:
    """Normalize accents and lowercase text for uniform token matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).lower()


def _clean_html_text(raw_html: str | None) -> str:
    """Decode HTML entities and strip markup tags into clean plain text."""
    if not raw_html:
        return ""
    unescaped = html.unescape(raw_html)
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


def _extract_ph_ddo(html_text: str) -> dict[str, Any]:
    """Parse the embedded `phApp.ddo` JSON payload from Phenom HTML."""
    marker = "phApp.ddo = "
    idx = html_text.find(marker)
    if idx == -1:
        return {}
    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(html_text[idx + len(marker) :])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class ThalesScraper(BaseScraper):
    """Scraper extracting local IT/cyber/AI roles from Thales Phenom portal.

    Attributes:
        company_name: Display employer name.
        allowed_locations: Target local cities and department tokens.
        timeout_seconds: HTTP timeout per request.
        location_matcher: Shared geo validator with 25 km radius check.
        max_pages: Maximum SSR pages fetched per area query.
    """

    def __init__(
        self,
        company_name: str = "Thales",
        allowed_locations: list[str] | None = None,
        timeout_seconds: float = 15.0,
        location_matcher: LocationMatcher | None = None,
        max_pages: int = DEFAULT_THALES_MAX_PAGES,
        area_queries: tuple[str, ...] = DEFAULT_THALES_AREA_QUERIES,
    ) -> None:
        """Initialize the Thales Phenom scraper.

        Args:
            company_name: Employer name shown in records.
            allowed_locations: Whitelisted local towns/tags.
            timeout_seconds: Request timeout in seconds.
            location_matcher: Shared geo matcher with 25 km distance check.
            max_pages: Pagination depth per area search.
            area_queries: Geographic search terms queried on Phenom SSR.
        """
        super().__init__(name="thales")
        self.company_name = company_name
        self.allowed_locations = [
            loc.lower()
            for loc in (allowed_locations or list(DEFAULT_THALES_ALLOWED_LOCATIONS))
        ]
        self.timeout_seconds = timeout_seconds
        self.location_matcher = location_matcher or LocationMatcher()
        self.max_pages = max_pages
        self.area_queries = area_queries

    @staticmethod
    def _matches_keywords_or_it(
        text: str,
        keywords: list[str],
    ) -> bool:
        """Check whether candidate text matches search keywords or IT signals."""
        norm_text = _strip_accents_lower(text)
        for kw in keywords:
            norm_kw = _strip_accents_lower(kw).strip()
            if norm_kw and norm_kw in norm_text:
                return True

        for token in DEFAULT_THALES_IT_TOKENS:
            escaped = re.escape(token)
            pattern = (
                rf"(?<![a-z0-9]){escaped}"
                if token.endswith(("developp", "algorith"))
                else rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
            )
            if re.search(pattern, norm_text):
                return True
        return False

    async def _fetch_job_detail(
        self,
        client: httpx.AsyncClient,
        job_seq_no: str,
    ) -> dict[str, Any]:
        """Fetch full job description and ATS apply URL from job detail SSR."""
        detail_url = THALES_JOB_URL_TEMPLATE.format(job_seq_no=job_seq_no)
        try:
            response = await client.get(detail_url)
            if response.status_code != 200:
                return {}
            ddo = _extract_ph_ddo(response.text)
            job_data = ddo.get("jobDetail", {}).get("data", {}).get("job", {})
            return job_data if isinstance(job_data, dict) else {}
        except Exception:
            return {}

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Query Thales Phenom search SSR for local Bouches-du-Rhone openings.

        Args:
            keywords: Target job title or technology keywords.
            locations: Target location filters from search_criteria.txt.
            limit: Maximum count of listings to emit.

        Yields:
            RawJobListing: Normalized Thales job offer with full description.
        """
        yielded_count = 0
        seen_seq_ids: set[str] = set()
        merged_locations = list(
            dict.fromkeys(
                self.allowed_locations + [loc.lower() for loc in locations]
            )
        )

        async with httpx.AsyncClient(
            headers=DEFAULT_THALES_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for area_query in self.area_queries:
                if yielded_count >= limit:
                    break

                for page_idx in range(self.max_pages):
                    if yielded_count >= limit:
                        break

                    offset = page_idx * DEFAULT_THALES_PAGE_SIZE
                    search_url = (
                        f"{THALES_SEARCH_URL}?keywords={quote_plus(area_query)}"
                        f"&from={offset}"
                    )
                    try:
                        response = await client.get(search_url)
                        if response.status_code != 200:
                            break

                        ddo = _extract_ph_ddo(response.text)
                        eager = ddo.get("eagerLoadRefineSearch", {})
                        data = eager.get("data", {})
                        postings = data.get("jobs", [])
                        if not isinstance(postings, list) or not postings:
                            break

                        for item in postings:
                            if yielded_count >= limit:
                                break
                            if not isinstance(item, dict):
                                continue

                            job_seq_no = str(
                                item.get("jobSeqNo") or item.get("jobId") or ""
                            ).strip()
                            title = str(item.get("title", "")).strip()
                            if not job_seq_no or not title:
                                continue
                            if job_seq_no in seen_seq_ids:
                                continue
                            seen_seq_ids.add(job_seq_no)

                            if is_excluded_title(title):
                                continue

                            loc_text = str(
                                item.get("cityStateCountry")
                                or item.get("location")
                                or item.get("city")
                                or ""
                            ).strip()
                            if not await self.location_matcher.is_location_allowed(
                                raw_location=loc_text,
                                extra_locations=merged_locations,
                                client=client,
                            ):
                                continue

                            teaser = str(item.get("descriptionTeaser", ""))
                            category = str(item.get("category", ""))
                            pre_text = f"{title} {teaser} {category}"
                            if not self._matches_keywords_or_it(
                                pre_text, keywords
                            ):
                                continue

                            # Récupération de la fiche complète Phenom
                            detail = await self._fetch_job_detail(
                                client=client,
                                job_seq_no=job_seq_no,
                            )
                            full_html_desc = str(
                                detail.get("description") or teaser or title
                            )
                            clean_desc = _clean_html_text(full_html_desc)
                            detail_url = THALES_JOB_URL_TEMPLATE.format(
                                job_seq_no=job_seq_no
                            )
                            apply_url = str(detail.get("applyUrl") or "").strip()

                            composed_desc = (
                                f"{title} chez {self.company_name}\n"
                                f"Lieu : {loc_text} | "
                                f"Catégorie : {category or 'Ingénierie / Tech'}\n\n"
                                f"Description de l'offre :\n"
                                f"{clean_desc or title}"
                            )

                            payload = dict(item)
                            payload["is_easy_apply"] = False
                            payload["location"] = loc_text
                            if apply_url:
                                payload["apply_url"] = apply_url

                            yield RawJobListing(
                                platform=self.name,
                                company=self.company_name,
                                title=title,
                                url=detail_url,
                                description=composed_desc,
                                location=loc_text,
                                raw_payload=payload,
                            )
                            yielded_count += 1

                        total_hits = int(eager.get("totalHits") or 0)
                        if (
                            offset + DEFAULT_THALES_PAGE_SIZE >= total_hits
                            or len(postings) < DEFAULT_THALES_PAGE_SIZE
                        ):
                            break

                    except Exception as err:
                        print(
                            f"⚠️ [{self.name}] Erreur sur '{area_query}' "
                            f"(offset={offset}) : {err}"
                        )
                        break
