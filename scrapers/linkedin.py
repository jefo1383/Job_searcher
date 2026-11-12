"""Hybrid scraper for LinkedIn Jobs combining guest API and Playwright fallback."""

import asyncio
from collections.abc import AsyncIterator
import html
import json
from pathlib import Path
import re
from urllib.parse import quote_plus, urljoin

import httpx
from playwright.async_api import BrowserContext, async_playwright

from config.settings import get_settings
from core.location_matcher import LocationMatcher, normalize_geo_text
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# En-têtes HTTP simulant un navigateur régulier pour l'API syndication
DEFAULT_LINKEDIN_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

LINKEDIN_GUEST_SEARCH_URL: str = (
    "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
)
LINKEDIN_GUEST_DETAIL_URL: str = (
    "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting"
)


def _clean_text(raw_text: str | None) -> str:
    """Strip HTML markup and collapse whitespace."""
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    clean_text = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", clean_text).strip()


def _extract_balanced_div(html_text: str, marker_regex: str) -> str:
    """Extract inner HTML of a div matching marker_regex, respecting nesting.

    Args:
        html_text: Full HTML document string.
        marker_regex: Regex matching the opening div tag.

    Returns:
        str: Inner HTML up to the matching closing div tag, or empty string.
    """
    open_match = re.search(marker_regex, html_text, re.IGNORECASE | re.DOTALL)
    if not open_match:
        return ""

    start_idx = open_match.end()
    depth = 1
    tag_pattern = re.compile(r"<div\b[^>]*>|</div>", re.IGNORECASE)
    for tag_match in tag_pattern.finditer(html_text, pos=start_idx):
        tag_str = tag_match.group(0).lower()
        if tag_str.startswith("</div"):
            depth -= 1
            if depth == 0:
                return html_text[start_idx:tag_match.start()]
        else:
            depth += 1

    # Repli borné si la balise fermante est absente ou tronquée
    return html_text[start_idx:start_idx + 12000]


def _extract_jsonld_description(html_text: str) -> str:
    """Extract job description from embedded JSON-LD schema if available."""
    for raw_json in re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        re.DOTALL | re.IGNORECASE,
    ):
        try:
            payload = json.loads(raw_json.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("description"), str
        ):
            return _clean_text(payload["description"])
    return ""


class LinkedInScraper(BaseScraper):
    """Hybrid scraper for LinkedIn Jobs supporting Easy Apply and external ATS.

    Attributes:
        user_data_dir: Persistent Chrome profile folder.
        headless: Whether browser fallback executes in headless mode.
        time_filter: LinkedIn date filter code (r604800 = past week).
        timeout_seconds: Request timeout for network calls.
        detail_delay_seconds: Polite pause before each job detail HTTP fetch.
        max_detail_retries: Retry attempts when LinkedIn returns HTTP 429/503.
        location_matcher: Shared geographic filter with 25km radius check.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        headless: bool = True,
        time_filter: str = "r604800",
        timeout_seconds: float = 15.0,
        detail_delay_seconds: float = 0.65,
        max_detail_retries: int = 2,
        location_matcher: LocationMatcher | None = None,
    ) -> None:
        """Initialize LinkedIn scraper.

        Args:
            user_data_dir: Optional custom Chrome user profile directory.
            headless: Display mode for browser fallback.
            time_filter: LinkedIn time window parameter.
            timeout_seconds: Max network wait duration.
            detail_delay_seconds: Delay between sequential detail calls.
            max_detail_retries: Retry count on HTTP 429 throttling.
            location_matcher: Shared geo matcher for dept 13 filtering.
        """
        super().__init__(name="linkedin")
        settings = get_settings()
        self.user_data_dir = user_data_dir or settings.chrome_user_data_dir
        self.headless = headless
        self.time_filter = time_filter
        self.timeout_seconds = timeout_seconds
        self.detail_delay_seconds = detail_delay_seconds
        self.max_detail_retries = max_detail_retries
        self.location_matcher = location_matcher or LocationMatcher()

    @staticmethod
    def _normalize_locations(
        locations: list[str], max_locations: int = 3
    ) -> list[str]:
        """Deduplicate and condense locations to avoid LinkedIn rate limits.

        Args:
            locations: Raw configured location list.
            max_locations: Max distinct queries per keyword (tunable).

        Returns:
            list[str]: Compact list of LinkedIn-compatible locations.
        """
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in locations:
            loc = raw.strip()
            if not loc:
                continue
            if loc == "13":
                loc = "Bouches-du-Rhône"
            elif loc.lower() == "télétravail":
                loc = "Remote"
            key = loc.lower()
            if key not in seen:
                seen.add(key)
                normalized.append(loc)
            if len(normalized) >= max_locations:
                break
        return normalized or ["Bouches-du-Rhône"]

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Scrape LinkedIn jobs across keywords and locations.

        Attempts high-speed guest syndication first, falling back to
        Playwright persistent browser session if restricted or throttled.

        Args:
            keywords: Target job keywords.
            locations: Geographic search terms.
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Normalized scraped job post.
        """
        yielded_count = 0
        target_locations = self._normalize_locations(locations)
        seen_ids: set[str] = set()
        consecutive_throttles = 0
        # Seuil de pauses 429 consécutives avant arrêt du cycle (modifiable)
        max_consecutive_throttles = 3

        for keyword in keywords:
            if yielded_count >= limit:
                break
            if consecutive_throttles >= max_consecutive_throttles:
                break

            for loc in target_locations:
                if yielded_count >= limit:
                    break
                if consecutive_throttles >= max_consecutive_throttles:
                    break

                guest_failed = False
                extracted_for_pair = 0

                # 1. Tentative rapide via l'API publique de syndication
                try:
                    async for raw in self._scrape_guest_api(
                        keyword=keyword,
                        location=loc,
                        limit=limit - yielded_count,
                        seen_ids=seen_ids,
                        allowed_locations=locations,
                    ):
                        yield raw
                        yielded_count += 1
                        extracted_for_pair += 1
                        consecutive_throttles = 0
                        if yielded_count >= limit:
                            break

                except Exception as err:
                    consecutive_throttles += 1
                    print(
                        f"⚠️ [LinkedIn] Guest API restricted for '{keyword}' "
                        f"in '{loc}': {err}. Bascule vers Playwright..."
                    )
                    guest_failed = True

                # 2. Repli Playwright uniquement sur le premier blocage détecté
                if (
                    guest_failed
                    and consecutive_throttles == 1
                    and extracted_for_pair == 0
                    and yielded_count < limit
                ):
                    try:
                        async for raw in self._scrape_playwright(
                            keyword=keyword,
                            location=loc,
                            limit=limit - yielded_count,
                        ):
                            yield raw
                            yielded_count += 1
                            consecutive_throttles = 0
                            if yielded_count >= limit:
                                break
                    except Exception as play_err:
                        print(
                            f"⚠️ [LinkedIn] Playwright fallback error for "
                            f"'{keyword}': {play_err}"
                        )

                await asyncio.sleep(0.5)

    async def _scrape_guest_api(
        self,
        keyword: str,
        location: str,
        limit: int = 20,
        seen_ids: set[str] | None = None,
        allowed_locations: list[str] | None = None,
    ) -> AsyncIterator[RawJobListing]:
        """Query LinkedIn guest syndication endpoint without browser overhead."""
        params: dict[str, str | int] = {
            "keywords": keyword,
            "location": location,
            "f_TPR": self.time_filter,
            "start": 0,
        }
        tracked_ids = seen_ids if seen_ids is not None else set()

        async with httpx.AsyncClient(
            headers=DEFAULT_LINKEDIN_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            res = await client.get(LINKEDIN_GUEST_SEARCH_URL, params=params)
            if res.status_code in (429, 403):
                raise RuntimeError(f"HTTP {res.status_code} Throttled")
            if res.status_code != 200:
                return

            html_text = res.text
            card_blocks = re.findall(
                r"<li[^>]*>(.*?)</li>", html_text, re.DOTALL | re.IGNORECASE
            )

            yielded_here = 0
            for block in card_blocks:
                if yielded_here >= limit:
                    break

                title_match = re.search(
                    r'<h3[^>]*class=["\'][^"\']*base-search-card__title[^"\']*["\']'
                    r'[^>]*>\s*(.*?)\s*</h3>',
                    block,
                    re.DOTALL,
                )
                company_match = re.search(
                    r'<h4[^>]*class=["\'][^"\']*base-search-card__subtitle'
                    r'[^"\']*["\'][^>]*>\s*<a[^>]*>(.*?)</a>',
                    block,
                    re.DOTALL,
                )
                if not company_match:
                    company_match = re.search(
                        r'<h4[^>]*class=["\'][^"\']*base-search-card__subtitle'
                        r'[^"\']*["\'][^>]*>\s*(.*?)\s*</h4>',
                        block,
                        re.DOTALL,
                    )
                loc_match = re.search(
                    r'<span[^>]*class=["\'][^"\']*job-search-card__location'
                    r'[^"\']*["\'][^>]*>\s*(.*?)\s*</span>',
                    block,
                    re.DOTALL,
                )
                link_match = re.search(
                    r'<a[^>]*href=["\'](https://[^"\']+/jobs/view/[^"\']+)["\']',
                    block,
                )

                if not title_match or not link_match:
                    continue

                raw_title = _clean_text(title_match.group(1))
                # Court-circuit avant l'appel HTTP de détail si titre bloquant
                if is_excluded_title(raw_title):
                    continue
                company = (
                    _clean_text(company_match.group(1))
                    if company_match
                    else "Entreprise"
                )
                job_loc = _clean_text(loc_match.group(1)) if loc_match else ""
                raw_url = link_match.group(1).split("?")[0].strip()

                job_id_match = re.search(r"(\d{8,})", raw_url)
                if not job_id_match:
                    continue
                job_id = job_id_match.group(1)
                if job_id in tracked_ids:
                    continue

                # Filtrage géographique partagé (critères + rayon 25 km dans le 13)
                norm_title = normalize_geo_text(raw_title)
                title_is_remote = any(
                    tok in norm_title
                    for tok in ("remote", "teletravail", "distanciel")
                )
                if job_loc and not title_is_remote:
                    is_allowed = await self.location_matcher.is_location_allowed(
                        raw_location=job_loc,
                        extra_locations=allowed_locations,
                        client=client,
                    )
                    if not is_allowed:
                        continue

                tracked_ids.add(job_id)
                canonical_url = f"https://www.linkedin.com/jobs/view/{job_id}"

                # Pause de courtoisie pour éviter le bridage rafale 429 sur les détails
                if self.detail_delay_seconds > 0:
                    await asyncio.sleep(self.detail_delay_seconds)

                # Récupération détaillée de l'offre (description et type de candidature)
                detail_url = f"{LINKEDIN_GUEST_DETAIL_URL}/{job_id}"
                desc, is_easy, apply_url = await self._fetch_guest_job_detail(
                    client=client,
                    detail_url=detail_url,
                )

                yield RawJobListing(
                    platform=self.name,
                    company=company,
                    title=raw_title,
                    url=canonical_url,
                    description=desc or raw_title,
                    location=job_loc,
                    raw_payload={
                        "job_id": job_id,
                        "is_easy_apply": is_easy,
                        "apply_url": apply_url or canonical_url,
                    },
                )
                yielded_here += 1

    async def _fetch_guest_job_detail(
        self,
        client: httpx.AsyncClient,
        detail_url: str,
    ) -> tuple[str, bool, str | None]:
        """Fetch job description and identify if Easy Apply or external ATS."""
        req_headers = {"Referer": "https://www.linkedin.com/jobs/search"}
        text = ""
        for attempt in range(self.max_detail_retries + 1):
            try:
                res = await client.get(detail_url, headers=req_headers)
                if res.status_code == 200:
                    text = res.text
                    break
                if (
                    res.status_code in (429, 503)
                    and attempt < self.max_detail_retries
                ):
                    # Backoff progressif en cas de limitation 429 sur la fiche
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                print(
                    f"⚠️ [LinkedIn] Détail ignoré (HTTP {res.status_code}) "
                    f"sur {detail_url}"
                )
                return "", False, None
            except Exception:
                if attempt < self.max_detail_retries:
                    await asyncio.sleep(1.0)
                    continue
                return "", False, None

        if not text:
            return "", False, None

        # Extraction robuste tolérant les <div> imbriqués
        inner_markup = _extract_balanced_div(
            text,
            r'<div[^>]*class=["\'][^"\']*show-more-less-html__markup[^"\']*["\'][^>]*>',
        )
        clean_desc = _clean_text(inner_markup)

        if not clean_desc:
            desc_section = re.search(
                r'<section[^>]*class=["\'][^"\']*description[^"\']*["\']'
                r"[^>]*>(.*?)</section>",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if desc_section:
                clean_desc = _clean_text(desc_section.group(1))

        if not clean_desc:
            clean_desc = _extract_jsonld_description(text)

        # Détection de lien externe vers ATS (prioritaire sur Easy Apply)
        apply_url: str | None = None
        ext_match = re.search(
            r'href=["\'](https://[^"\']+(?:applyUrl|externalApply)[^"\']+)["\']',
            text,
        )
        if not ext_match:
            ext_match = re.search(
                r'<code[^>]*id=["\']applyUrl["\'][^>]*>\s*'
                r'<!--\s*"(https?://[^"]+)"\s*-->',
                text,
            )
        if not ext_match:
            ext_match = re.search(
                r'<a[^>]+class=["\'][^"\']*apply-button[^"\']*["\'][^>]+'
                r'href=["\'](https://[^"\']+)["\']',
                text,
            )
        if ext_match:
            apply_url = html.unescape(ext_match.group(1))

        # Détection de Candidature Simplifiée (Easy Apply)
        has_offsite_icon = "offsite-apply" in text or (
            apply_url is not None and "linkedin.com/jobs" not in apply_url
        )
        is_easy = bool(
            not has_offsite_icon
            and (
                "apply-button--easy" in text
                or "Candidature simplifiée" in text
                or "Easy Apply" in text
            )
        )

        return clean_desc, is_easy, apply_url

    async def _scrape_playwright(
        self,
        keyword: str,
        location: str,
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Scrape LinkedIn jobs via persistent Playwright Chrome session."""
        profile_dir = self.user_data_dir
        if not profile_dir.exists():
            try:
                profile_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                profile_dir = Path("data/chrome_profile")
                profile_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
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

            try:
                page = await context.new_page()
                search_url = (
                    f"https://www.linkedin.com/jobs/search/?"
                    f"keywords={quote_plus(keyword)}&"
                    f"location={quote_plus(location)}&"
                    f"f_TPR={self.time_filter}"
                )

                await page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                await page.wait_for_timeout(2500)

                # Extraction des cartes d'offres affichées dans la liste
                job_cards = await page.locator(".job-card-container").all()
                count = 0

                for card in job_cards:
                    if count >= limit:
                        break

                    try:
                        title_el = card.locator(".job-card-list__title").first
                        if not await title_el.count():
                            continue
                        title_text = _clean_text(await title_el.text_content())
                        if is_excluded_title(title_text):
                            continue

                        company_el = card.locator(
                            ".job-card-container__primary-description"
                        ).first
                        company_text = (
                            _clean_text(await company_el.text_content())
                            if await company_el.count()
                            else "Entreprise"
                        )

                        loc_el = card.locator(
                            ".job-card-container__metadata-item"
                        ).first
                        loc_text = (
                            _clean_text(await loc_el.text_content())
                            if await loc_el.count()
                            else ""
                        )

                        # Clic sur la carte pour charger le volet de droite
                        await card.click()
                        await page.wait_for_timeout(1000)

                        # Détection du bouton de candidature
                        apply_btn = page.locator(".jobs-apply-button").first
                        is_easy = False
                        apply_url = None

                        if await apply_btn.count():
                            btn_text = await apply_btn.text_content() or ""
                            if (
                                "candidature simplifiée" in btn_text.lower()
                                or "easy apply" in btn_text.lower()
                            ):
                                is_easy = True
                            else:
                                is_easy = False
                                tag = await apply_btn.evaluate(
                                    "el => el.tagName.toLowerCase()"
                                )
                                if tag == "a":
                                    apply_url = await apply_btn.get_attribute(
                                        "href"
                                    )

                        # Extraction du corps de description
                        desc_el = page.locator(
                            ".jobs-description__content"
                        ).first
                        desc_text = (
                            _clean_text(await desc_el.text_content())
                            if await desc_el.count()
                            else title_text
                        )

                        # Construction de l'URL
                        card_url = page.url
                        link_tag = card.locator("a").first
                        if await link_tag.count():
                            href = await link_tag.get_attribute("href")
                            if href:
                                card_url = urljoin(page.url, href).split("?")[
                                    0
                                ]

                        yield RawJobListing(
                            platform=self.name,
                            company=company_text,
                            title=title_text,
                            url=card_url,
                            description=desc_text,
                            location=loc_text,
                            raw_payload={
                                "is_easy_apply": is_easy,
                                "apply_url": apply_url or card_url,
                            },
                        )
                        count += 1

                    except Exception:
                        continue

            finally:
                await context.close()
