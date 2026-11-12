"""Workday Candidate Experience Services (CXS) REST API scraper."""

import html
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.location_matcher import LocationMatcher
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# Nombre maximal de pages Workday inspectées par mot-clé (réglable)
DEFAULT_MAX_WORKDAY_PAGES: int = 3
DEFAULT_WORKDAY_PAGE_SIZE: int = 20

# En-têtes HTTP simulant un navigateur standard
DEFAULT_WORKDAY_HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


class WorkdayScraper(BaseScraper):
    """Reusable scraper for corporate career portals powered by Workday CXS.

    Communicates directly with the public Workday CXS REST endpoints,
    retrieving structured JSON listings and job descriptions without
    requiring a browser emulation layer.

    Attributes:
        company_name: Display name of the employing organization.
        tenant_domain: Domain name of the Workday tenant.
        company_path: First routing segment in the CXS URL.
        portal_path: Second routing segment representing the career site.
        allowed_locations: Whitelist of allowed location substrings.
        timeout_seconds: Network request timeout in seconds.
    """

    def __init__(
        self,
        name: str,
        company_name: str,
        tenant_domain: str,
        company_path: str,
        portal_path: str,
        allowed_locations: list[str] | None = None,
        timeout_seconds: float = 15.0,
        location_matcher: LocationMatcher | None = None,
        max_pages: int = DEFAULT_MAX_WORKDAY_PAGES,
    ) -> None:
        """Initialize the Workday scraper.

        Args:
            name: Platform identifier name.
            company_name: Hiring entity name.
            tenant_domain: FQDN of Workday tenant.
            company_path: CXS routing segment for company.
            portal_path: CXS routing segment for target site.
            allowed_locations: List of location keywords to retain.
            timeout_seconds: Max wait duration per HTTP call.
            location_matcher: Shared geo validator with 25km radius check.
            max_pages: Max CXS pages inspected per keyword before moving on.
        """
        super().__init__(name=name)
        self.company_name = company_name
        self.tenant_domain = tenant_domain
        self.company_path = company_path
        self.portal_path = portal_path
        self.allowed_locations = [
            loc.lower() for loc in (allowed_locations or [])
        ]
        self.timeout_seconds = timeout_seconds
        self.location_matcher = location_matcher or LocationMatcher()
        self.max_pages = max_pages

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Query Workday CXS API for matching positions.

        Args:
            keywords: Target role or technology keywords.
            locations: Secondary geographic search criteria.
            limit: Maximum count of listings to extract per search.

        Yields:
            RawJobListing: Normalized scraped job post.
        """
        search_endpoint = (
            f"https://{self.tenant_domain}/wday/cxs/"
            f"{self.company_path}/{self.portal_path}/jobs"
        )

        yielded_count = 0
        seen_urls: set[str] = set()
        async with httpx.AsyncClient(
            headers=DEFAULT_WORKDAY_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for keyword in keywords:
                if yielded_count >= limit:
                    break

                for page_idx in range(self.max_pages):
                    if yielded_count >= limit:
                        break

                    offset = page_idx * DEFAULT_WORKDAY_PAGE_SIZE
                    payload: dict[str, Any] = {
                        "appliedFacets": {},
                        "limit": DEFAULT_WORKDAY_PAGE_SIZE,
                        "offset": offset,
                        "searchText": keyword,
                    }

                    try:
                        response = await client.post(
                            search_endpoint,
                            json=payload,
                        )
                        if response.status_code != 200:
                            break

                        data = response.json()
                        postings = data.get("jobPostings", [])
                        if not postings:
                            break

                        for item in postings:
                            if yielded_count >= limit:
                                break

                            title = item.get("title", "").strip()
                            locations_text = item.get(
                                "locationsText", ""
                            ).strip()
                            external_path = item.get(
                                "externalPath", ""
                            ).strip()

                            if not title or not external_path:
                                continue

                            if is_excluded_title(title):
                                continue

                            public_url = (
                                f"https://{self.tenant_domain}/"
                                f"{self.portal_path}{external_path}"
                            )
                            if public_url in seen_urls:
                                continue

                            # Vérification géographique (critères + rayon 25 km)
                            if not await self.location_matcher.is_location_allowed(
                                raw_location=locations_text,
                                extra_locations=self.allowed_locations,
                                client=client,
                            ):
                                continue

                            seen_urls.add(public_url)

                            # Construction de l'URL de détail CXS
                            detail_endpoint = (
                                f"https://{self.tenant_domain}/wday/cxs/"
                                f"{self.company_path}/{self.portal_path}"
                                f"{external_path}"
                            )

                            # Récupération de la description détaillée
                            description = await self._fetch_job_description(
                                client=client,
                                detail_endpoint=detail_endpoint,
                            )

                            enriched_payload = dict(item)
                            enriched_payload["is_easy_apply"] = False

                            yield RawJobListing(
                                platform=self.name,
                                company=self.company_name,
                                title=title,
                                url=public_url,
                                description=description or title,
                                location=locations_text,
                                raw_payload=enriched_payload,
                            )
                            yielded_count += 1

                        total_available = int(data.get("total") or 0)
                        if (
                            offset + DEFAULT_WORKDAY_PAGE_SIZE
                            >= total_available
                            or len(postings) < DEFAULT_WORKDAY_PAGE_SIZE
                        ):
                            break

                    except Exception as err:
                        print(
                            f"⚠️ [{self.name}] Error querying keyword "
                            f"'{keyword}': {err}"
                        )
                        break

    def _is_location_allowed(self, location_str: str) -> bool:
        """Verify if a job's location matches target geographical filters.

        Args:
            location_str: Raw location text from Workday API.

        Returns:
            bool: True if matched or whitelist is empty, False otherwise.
        """
        if not self.allowed_locations:
            return True
        loc_lower = location_str.lower()
        return any(target in loc_lower for target in self.allowed_locations)

    async def _fetch_job_description(
        self,
        client: httpx.AsyncClient,
        detail_endpoint: str,
    ) -> str:
        """Fetch and sanitize the job description body from CXS detail API.

        Args:
            client: Active HTTP client session.
            detail_endpoint: Full detail REST URL.

        Returns:
            str: Plain text description with HTML tags stripped.
        """
        try:
            detail_res = await client.get(detail_endpoint)
            if detail_res.status_code != 200:
                return ""

            detail_data = detail_res.json()
            info = detail_data.get("jobPostingInfo", {})
            html_content = info.get("jobDescription", "")
            if not html_content:
                return ""

            # Décodage des entités HTML et suppression des balises
            unescaped = html.unescape(html_content)
            clean_text = re.sub(r"<[^>]+>", " ", unescaped)
            return re.sub(r"\s+", " ", clean_text).strip()

        except Exception:
            return ""


class AirbusScraper(WorkdayScraper):
    """Specialized Workday scraper for Airbus targeting PACA aerospace facilities."""

    def __init__(self) -> None:
        """Initialize Airbus scraper with Marignane, Vitrolles, and Aix filters."""
        super().__init__(
            name="airbus",
            company_name="Airbus",
            tenant_domain="ag.wd3.myworkdayjobs.com",
            company_path="ag",
            portal_path="Airbus",
            allowed_locations=[
                "marignane",
                "vitrolles",
                "aix",
                "marseille",
                "marseille area",
                "13",
                "remote",
                "télétravail",
            ],
        )


class FDJScraper(WorkdayScraper):
    """Specialized Workday scraper for Groupe FDJ (Française des Jeux)."""

    def __init__(self) -> None:
        """Initialize FDJ scraper with Vitrolles and PACA tech hub filters."""
        super().__init__(
            name="fdj",
            company_name="Groupe FDJ",
            tenant_domain="groupefdj.wd103.myworkdayjobs.com",
            company_path="groupefdj",
            portal_path="FDJ-UNITED",
            allowed_locations=[
                "vitrolles",
                "marseille",
                "aix",
                "13",
                "remote",
                "télétravail",
            ],
        )
