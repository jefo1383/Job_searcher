"""Scraper implementation for Welcome to the Jungle platform."""

from collections.abc import AsyncIterator
import html
import json
import re
from typing import Any

import httpx
from playwright.async_api import BrowserContext, async_playwright

from config.settings import get_settings
from core.location_matcher import LocationMatcher
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# URLs publiques WTTJ (récupération dynamique de la config et fiches)
WTTJ_ENV_URL: str = "https://www.welcometothejungle.com/api/env"
WTTJ_SEARCH_PAGE_URL: str = "https://www.welcometothejungle.com/fr/jobs"
WTTJ_JOB_API_TEMPLATE: str = (
    "https://api.welcometothejungle.com/api/v1/"
    "organizations/{org_slug}/jobs/{job_slug}"
)
DEFAULT_WTTJ_INDEXES: tuple[str, ...] = (
    "wk_cms_jobs_production_published_at_desc",
    "wk_cms_jobs_production",
)
CHROME_USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def _clean_html_text(raw_text: str | None) -> str:
    """Remove HTML tags, unescape entities, and collapse whitespace.

    Args:
        raw_text: Raw string possibly containing HTML markup.

    Returns:
        str: Cleaned plain text string.
    """
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


class WTTJScraper(BaseScraper):
    """Scraper targeting Welcome to the Jungle with dynamic Algolia key lookup.

    Fetches the current PUBLIC_ALGOLIA_API_KEY_CLIENT from WTTJ's /api/env
    endpoint on every execution, queries Algolia within the 25km radius,
    hydrates full offer details, and falls back to Playwright if needed.
    """

    def __init__(
        self,
        location_matcher: LocationMatcher | None = None,
        timeout_seconds: float = 15.0,
        headless: bool = True,
    ) -> None:
        """Initialize the Welcome to the Jungle scraper.

        Args:
            location_matcher: Shared geo validator with 25km radius check.
            timeout_seconds: HTTP/Playwright timeout duration in seconds.
            headless: Whether Playwright fallback runs headless.
        """
        super().__init__(name="wttj")
        self.location_matcher = location_matcher or LocationMatcher()
        self.timeout_seconds = timeout_seconds
        self.headless = headless

    @staticmethod
    def _extract_env_json(raw_script: str) -> dict[str, Any]:
        """Parse JSON object from WTTJ window.env script body.

        Args:
            raw_script: JavaScript text returned by /api/env.

        Returns:
            dict[str, Any]: Parsed environment dictionary or empty dict.
        """
        match = re.search(
            r"window\.env\s*=\s*(\{.*\})\s*;?",
            raw_script,
            re.DOTALL,
        )
        if not match:
            return {}
        try:
            data = json.loads(match.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    async def _fetch_live_algolia_credentials(
        self, client: httpx.AsyncClient
    ) -> tuple[str, str] | None:
        """Fetch live Algolia App ID and API key from WTTJ runtime env.

        Args:
            client: Reusable async HTTP client.

        Returns:
            tuple[str, str] | None: Pair of (app_id, api_key) or None.
        """
        headers = {
            "User-Agent": CHROME_USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Referer": "https://www.welcometothejungle.com/fr/jobs",
        }
        try:
            response = await client.get(WTTJ_ENV_URL, headers=headers)
            if response.status_code != 200:
                return None
            env_data = self._extract_env_json(response.text)
            app_id = str(
                env_data.get("PUBLIC_ALGOLIA_APPLICATION_ID", "")
            ).strip()
            api_key = str(
                env_data.get("PUBLIC_ALGOLIA_API_KEY_CLIENT", "")
            ).strip()
            if app_id and api_key:
                return app_id, api_key
        except httpx.HTTPError:
            return None
        return None

    async def _fetch_job_detail(
        self,
        client: httpx.AsyncClient,
        org_slug: str,
        job_slug: str,
    ) -> dict[str, Any]:
        """Fetch full WTTJ job offer details from public JSON API.

        Args:
            client: Reusable async HTTP client.
            org_slug: Organization URL slug.
            job_slug: Job offer URL slug.

        Returns:
            dict[str, Any]: Job detail dictionary or empty dict on error.
        """
        url = WTTJ_JOB_API_TEMPLATE.format(
            org_slug=org_slug, job_slug=job_slug
        )
        try:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": CHROME_USER_AGENT,
                    "Accept": "application/json",
                },
            )
            if resp.status_code != 200:
                return {}
            payload = resp.json()
            job_obj = payload.get("job", {})
            return job_obj if isinstance(job_obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_city(hit: dict[str, Any]) -> str:
        """Extract city/district string from an Algolia WTTJ hit."""
        offices = hit.get("offices")
        if isinstance(offices, list) and offices:
            first = offices[0]
            if isinstance(first, dict):
                city = str(first.get("city") or "").strip()
                district = str(first.get("district") or "").strip()
                if city and district:
                    return f"{city} ({district})"
                if city:
                    return city

        office = hit.get("office")
        if isinstance(office, dict):
            return str(office.get("city") or "").strip()
        return ""

    async def _build_listing(
        self,
        client: httpx.AsyncClient,
        hit: dict[str, Any],
    ) -> RawJobListing | None:
        """Normalize and hydrate a WTTJ Algolia hit into a RawJobListing."""
        org = hit.get("organization", {})
        org_name = (
            str(org.get("name", "Entreprise Inconnue")).strip()
            if isinstance(org, dict)
            else "Entreprise Inconnue"
        )
        org_slug = (
            str(org.get("slug", "")).strip() if isinstance(org, dict) else ""
        )
        job_title = str(hit.get("name", "")).strip()
        job_slug = str(hit.get("slug", "")).strip()

        if not org_slug or not job_slug or not job_title:
            return None

        if is_excluded_title(job_title):
            return None

        city_label = self._extract_city(hit)
        remote_mode = str(hit.get("remote") or "").lower()
        location_probe = (
            f"{city_label} remote"
            if remote_mode in ("full", "fulltime")
            else city_label
        )
        if not await self.location_matcher.is_location_allowed(
            raw_location=location_probe,
            client=client,
        ):
            return None

        job_url = (
            f"https://www.welcometothejungle.com/fr/"
            f"companies/{org_slug}/jobs/{job_slug}"
        )

        # Récupération de la fiche complète via l'API publique WTTJ
        detail = await self._fetch_job_detail(client, org_slug, job_slug)
        contract_type = str(
            detail.get("contract_type")
            or hit.get("contract_type")
            or ""
        ).strip()
        desc_clean = _clean_html_text(
            detail.get("description") or hit.get("description")
        )
        profile_clean = _clean_html_text(
            detail.get("profile") or hit.get("profile")
        )
        tools_list = detail.get("tools", [])
        tools_names = [
            str(t.get("name", "")).strip()
            for t in tools_list
            if isinstance(t, dict) and t.get("name")
        ]

        sections: list[str] = []
        if desc_clean:
            sections.append(f"Description :\n{desc_clean}")
        if profile_clean:
            sections.append(f"Profil recherché :\n{profile_clean}")
        if tools_names:
            sections.append(f"Outils / Stack : {', '.join(tools_names)}")

        full_desc = (
            f"{job_title} chez {org_name}\n"
            f"Lieu : {city_label or 'Non spécifié'} | "
            f"Contrat : {contract_type}\n\n"
            + "\n\n".join(sections)
        )

        raw_payload = dict(hit)
        raw_payload["is_easy_apply"] = False
        if contract_type:
            raw_payload["contract_type"] = contract_type
        apply_url = detail.get("apply_url")
        if isinstance(apply_url, str) and apply_url:
            raw_payload["apply_url"] = apply_url

        return RawJobListing(
            platform=self.name,
            company=org_name,
            title=job_title,
            url=job_url,
            description=full_desc,
            location=city_label,
            raw_payload=raw_payload,
        )

    async def _scrape_algolia(
        self,
        client: httpx.AsyncClient,
        app_id: str,
        api_key: str,
        keywords: list[str],
        limit: int,
        seen_urls: set[str],
    ) -> tuple[list[RawJobListing], bool]:
        """Query Algolia with dynamically retrieved public credentials.

        Args:
            client: Reusable async HTTP client.
            app_id: Current Algolia application ID.
            api_key: Current PUBLIC_ALGOLIA_API_KEY_CLIENT token.
            keywords: Search terms to query.
            limit: Maximum number of offers to collect.
            seen_urls: Set tracking deduplicated offer URLs.

        Returns:
            tuple[list[RawJobListing], bool]: Extracted listings and
                whether Algolia succeeded without auth/blocking failure.
        """
        headers = {
            "x-algolia-application-id": app_id,
            "x-algolia-api-key": api_key,
            "Origin": "https://www.welcometothejungle.com",
            "Referer": "https://www.welcometothejungle.com/",
            "Content-Type": "application/json",
            "User-Agent": CHROME_USER_AGENT,
        }
        lat, lon = self.location_matcher.residence_coords
        radius_meters = int(self.location_matcher.max_distance_km * 1000)

        listings: list[RawJobListing] = []
        active_index: str | None = None

        for keyword in keywords:
            if len(listings) >= limit:
                break

            indexes_to_try = (
                (active_index,) if active_index else DEFAULT_WTTJ_INDEXES
            )
            payload: dict[str, Any] = {
                "query": keyword,
                "aroundLatLng": f"{lat},{lon}",
                "aroundRadius": radius_meters,
                "hitsPerPage": min(max((limit - len(listings)) * 4, 15), 30),
            }

            response: httpx.Response | None = None
            for idx_name in indexes_to_try:
                algolia_url = (
                    f"https://{app_id.lower()}-dsn.algolia.net/1/"
                    f"indexes/{idx_name}/query"
                )
                try:
                    resp = await client.post(
                        algolia_url, headers=headers, json=payload
                    )
                    if resp.status_code == 404:
                        continue
                    response = resp
                    if resp.status_code == 200:
                        active_index = idx_name
                    break
                except httpx.HTTPError:
                    response = None
                    break

            if response is None:
                return listings, False

            if response.status_code in (401, 403):
                return listings, False
            if response.status_code != 200:
                continue

            hits = response.json().get("hits", [])
            for hit in hits:
                if len(listings) >= limit:
                    break
                if not isinstance(hit, dict):
                    continue
                built = await self._build_listing(client=client, hit=hit)
                if built is None or built.url in seen_urls:
                    continue
                seen_urls.add(built.url)
                listings.append(built)

        return listings, True

    async def _scrape_playwright_fallback(
        self,
        keywords: list[str],
        limit: int,
        seen_urls: set[str],
    ) -> list[RawJobListing]:
        """Fallback scraping WTTJ via Playwright session and network capture.

        Args:
            keywords: Search terms to query.
            limit: Max offers to return.
            seen_urls: Already collected URLs set.

        Returns:
            list[RawJobListing]: Listings gathered from Playwright or captured
                Algolia credentials.
        """
        settings = get_settings()
        profile_dir = settings.chrome_user_data_dir
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            profile_dir = settings.database_path.parent / "chrome_profile"
            profile_dir.mkdir(parents=True, exist_ok=True)

        results: list[RawJobListing] = []
        captured_creds: dict[str, str] = {}

        async with async_playwright() as p:
            try:
                context: BrowserContext = (
                    await p.chromium.launch_persistent_context(
                        user_data_dir=str(profile_dir),
                        headless=self.headless,
                        args=[
                            "--disable-blink-features=AutomationControlled",
                            "--no-sandbox",
                        ],
                        viewport={"width": 1280, "height": 800},
                    )
                )
            except Exception as err:
                print(f"⚠️ [WTTJScraper] Playwright fallback error: {err}")
                return results

            try:
                page = await context.new_page()

                def _capture_algolia_headers(req: Any) -> None:
                    if "algolia.net" in req.url:
                        req_headers = req.headers
                        key = req_headers.get("x-algolia-api-key", "")
                        app = req_headers.get(
                            "x-algolia-application-id", ""
                        )
                        if key and app:
                            captured_creds["app_id"] = app
                            captured_creds["api_key"] = key

                page.on("request", _capture_algolia_headers)

                first_kw = keywords[0] if keywords else "Python"
                resp = await page.goto(
                    f"{WTTJ_SEARCH_PAGE_URL}?query={first_kw}",
                    wait_until="domcontentloaded",
                    timeout=int(self.timeout_seconds * 1000),
                )
                if resp is not None and resp.status == 403:
                    print(
                        "⚠️ [WTTJScraper] Repli Playwright bloqué (HTTP 403)."
                    )
                    return results

                await page.wait_for_timeout(3500)

                # Si la session Playwright a intercepté la clé Algolia en direct
                if (
                    captured_creds.get("app_id")
                    and captured_creds.get("api_key")
                ):
                    async with httpx.AsyncClient(
                        timeout=self.timeout_seconds
                    ) as client:
                        listings, ok = await self._scrape_algolia(
                            client=client,
                            app_id=captured_creds["app_id"],
                            api_key=captured_creds["api_key"],
                            keywords=keywords,
                            limit=limit,
                            seen_urls=seen_urls,
                        )
                        if ok and listings:
                            return listings

                # Sinon extraction directe des liens d'offres dans le DOM
                links = await page.locator(
                    "a[href*='/fr/companies/'][href*='/jobs/']"
                ).all()
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds
                ) as client:
                    for link in links:
                        if len(results) >= limit:
                            break
                        href = await link.get_attribute("href")
                        if not href:
                            continue
                        match = re.search(
                            r"/fr/companies/([^/]+)/jobs/([^/?#]+)", href
                        )
                        if not match:
                            continue
                        org_slug, job_slug = match.groups()
                        job_url = (
                            f"https://www.welcometothejungle.com/fr/"
                            f"companies/{org_slug}/jobs/{job_slug}"
                        )
                        if job_url in seen_urls:
                            continue
                        detail = await self._fetch_job_detail(
                            client, org_slug, job_slug
                        )
                        if not detail:
                            continue
                        title = str(detail.get("name", "")).strip()
                        if not title or is_excluded_title(title):
                            continue
                        office = detail.get("office", {})
                        city = (
                            str(office.get("city", "")).strip()
                            if isinstance(office, dict)
                            else ""
                        )
                        if not await self.location_matcher.is_location_allowed(
                            raw_location=city, client=client
                        ):
                            continue
                        org_info = detail.get("organization", {})
                        org_name = (
                            str(org_info.get("name", org_slug)).strip()
                            if isinstance(org_info, dict)
                            else org_slug
                        )
                        desc = _clean_html_text(detail.get("description"))
                        prof = _clean_html_text(detail.get("profile"))
                        contract = str(
                            detail.get("contract_type") or ""
                        ).strip()
                        seen_urls.add(job_url)
                        results.append(
                            RawJobListing(
                                platform=self.name,
                                company=org_name,
                                title=title,
                                url=job_url,
                                description=f"{desc}\n\n{prof}".strip()
                                or title,
                                location=city,
                                raw_payload={
                                    "contract_type": contract,
                                    "is_easy_apply": False,
                                },
                            )
                        )
            except Exception as err:
                print(f"⚠️ [WTTJScraper] Playwright fallback error: {err}")
            finally:
                await context.close()

        return results

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch listings from Welcome to the Jungle matching criteria.

        Args:
            keywords: Target keywords to query.
            locations: Geographical filters (handled via LocationMatcher).
            limit: Maximum count of listings to return.

        Yields:
            RawJobListing: Normalized raw job post items.
        """
        seen_urls: set[str] = set()
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, follow_redirects=True
        ) as client:
            creds = await self._fetch_live_algolia_credentials(client)
            if creds is not None:
                app_id, api_key = creds
                listings, algolia_ok = await self._scrape_algolia(
                    client=client,
                    app_id=app_id,
                    api_key=api_key,
                    keywords=keywords,
                    limit=limit,
                    seen_urls=seen_urls,
                )
                for item in listings:
                    yield item
                if algolia_ok or len(listings) >= limit:
                    return

        # Repli Playwright si /api/env ou Algolia a échoué
        print(
            "ℹ️ [WTTJScraper] Bascule vers le repli Playwright "
            "suite à l'échec Algolia."
        )
        fallback_items = await self._scrape_playwright_fallback(
            keywords=keywords,
            limit=limit,
            seen_urls=seen_urls,
        )
        for item in fallback_items:
            yield item
