"""Scraper implementation for Orange Cyberdefense career opportunities."""

import html
import json
import re
from collections.abc import AsyncIterator
from urllib.parse import urljoin

import httpx

from core.location_matcher import LocationMatcher
from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# En-têtes HTTP simulant une requête de navigation standard
DEFAULT_ORANGE_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Points d'accès du portail Orange Jobs (Phenom People)
ORANGE_SEARCH_URL: str = "https://orange.jobs/fr/fr/search-results"
ORANGE_JOB_BASE_URL: str = "https://orange.jobs/fr/fr/job"


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
    # Double décodage si les balises HTML étaient encodées (&lt;p&gt;)
    if "<" not in unescaped and "&lt;" in raw_html:
        unescaped = html.unescape(unescaped)
    clean_text = re.sub(r"<[^>]+>", " ", unescaped)
    clean_text = html.unescape(clean_text)
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
        location_matcher: LocationMatcher | None = None,
    ) -> None:
        """Initialize the Orange Cyberdefense scraper.

        Args:
            company_name: Name of hiring company.
            entity: Filter segment for target Orange business unit.
            allowed_locations: List of target location tokens.
            timeout_seconds: Network call timeout in seconds.
            location_matcher: Shared geo validator with 25km radius check.
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
        self.location_matcher = location_matcher or LocationMatcher()

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Query Orange jobs Phenom People portal for matching positions.

        Args:
            keywords: Target role or technology keywords.
            locations: Secondary geographic search criteria.
            limit: Maximum count of listings to extract per search.

        Yields:
            RawJobListing: Normalized scraped job post.
        """
        yielded_count = 0
        async with httpx.AsyncClient(
            headers=DEFAULT_ORANGE_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for keyword in keywords:
                if yielded_count >= limit:
                    break

                try:
                    response = await client.get(
                        ORANGE_SEARCH_URL,
                        params={"keywords": keyword},
                    )
                    if response.status_code != 200:
                        continue

                    # Décodage de l'objet JavaScript Phenom People DDO
                    idx = response.text.find("phApp.ddo = ")
                    if idx == -1:
                        continue

                    start = idx + len("phApp.ddo = ")
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(response.text[start:])

                    eager = data.get("eagerLoadRefineSearch", {})
                    jobs = eager.get("data", {}).get("jobs", [])

                    for item in jobs:
                        if yielded_count >= limit:
                            break

                        title = item.get("title", "").strip()
                        raw_loc = item.get("city", "").strip()
                        job_id = str(item.get("jobId", "")).strip()

                        if not title or not job_id:
                            continue

                        # Vérification de la composante Cyber / SOC
                        category = item.get("category", "")
                        org = item.get("organization", "")
                        is_cyber = (
                            "cyber" in title.lower()
                            or "soc" in title.lower()
                            or "secops" in title.lower()
                            or "sécurité" in title.lower()
                            or "cyber" in keyword.lower()
                            or "cyber" in category.lower()
                            or "cyber" in org.lower()
                        )
                        if not is_cyber:
                            continue

                        # Vérification de la zone géographique (rayon 25 km)
                        if not await self.location_matcher.is_location_allowed(
                            raw_location=raw_loc,
                            extra_locations=self.allowed_locations,
                            client=client,
                        ):
                            continue

                        apply_url = item.get("applyUrl")
                        job_url_rel = item.get("jobUrl")
                        if job_url_rel:
                            public_url = urljoin("https://orange.jobs", job_url_rel)
                        else:
                            public_url = f"{ORANGE_JOB_BASE_URL}/{job_id}"

                        raw_contract = str(
                            item.get("contractType") or item.get("type") or ""
                        ).strip()
                        teaser = item.get("descriptionTeaser") or title
                        desc_body = await self._fetch_full_description(
                            client=client,
                            job_url=public_url,
                            fallback_text=teaser,
                        )
                        desc = (
                            f"Contrat : {raw_contract}\n{desc_body}"
                            if raw_contract
                            else desc_body
                        )

                        yield RawJobListing(
                            platform=self.name,
                            company=self.company_name,
                            title=title,
                            url=public_url,
                            description=desc,
                            location=raw_loc,
                            raw_payload={
                                "job_id": job_id,
                                "contract_type": raw_contract,
                                "is_easy_apply": False,
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

    async def _fetch_full_description(
        self,
        client: httpx.AsyncClient,
        job_url: str,
        fallback_text: str,
    ) -> str:
        """Fetch full offer text from Orange Jobs detail page.

        Args:
            client: Active HTTPX client session.
            job_url: Public Orange Jobs detail URL.
            fallback_text: Short teaser text if detail fetch fails.

        Returns:
            str: Full plain-text job description, or fallback teaser.
        """
        try:
            res = await client.get(job_url)
            if res.status_code != 200:
                return fallback_text

            # 1. Extraction depuis l'objet embarqué Phenom People DDO
            idx = res.text.find("phApp.ddo = ")
            if idx != -1:
                decoder = json.JSONDecoder()
                ddo, _ = decoder.raw_decode(res.text[idx + len("phApp.ddo = ") :])
                job_data = ddo.get("jobDetail", {}).get("data", {}).get("job", {})
                raw_desc = (
                    job_data.get("descriptionFr")
                    or job_data.get("description")
                    or ""
                )
                cleaned = _clean_html(raw_desc)
                if len(cleaned) > len(fallback_text):
                    return cleaned

            # 2. Repli JSON-LD JobPosting
            ld_blocks = re.findall(
                r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>'
                r"(.*?)</script>",
                res.text,
                re.DOTALL | re.IGNORECASE,
            )
            for block in ld_blocks:
                if "JobPosting" not in block:
                    continue
                parsed = json.loads(block)
                if isinstance(parsed, dict) and parsed.get("description"):
                    cleaned = _clean_html(str(parsed["description"]))
                    if len(cleaned) > len(fallback_text):
                        return cleaned
        except Exception:
            pass
        return fallback_text

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
