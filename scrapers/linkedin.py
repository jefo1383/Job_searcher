"""Hybrid scraper for LinkedIn Jobs combining guest API and Playwright fallback."""

from collections.abc import AsyncIterator
import html
from pathlib import Path
import re
from urllib.parse import quote_plus, urljoin

import httpx
from playwright.async_api import BrowserContext, async_playwright

from config.settings import get_settings
from scrapers.base import BaseScraper
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


class LinkedInScraper(BaseScraper):
    """Hybrid scraper for LinkedIn Jobs supporting Easy Apply and external ATS.

    Attributes:
        user_data_dir: Persistent Chrome profile folder.
        headless: Whether browser fallback executes in headless mode.
        time_filter: LinkedIn date filter code (r604800 = past week).
        timeout_seconds: Request timeout for network calls.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        headless: bool = True,
        time_filter: str = "r604800",
        timeout_seconds: float = 15.0,
    ) -> None:
        """Initialize LinkedIn scraper.

        Args:
            user_data_dir: Optional custom Chrome user profile directory.
            headless: Display mode for browser fallback.
            time_filter: LinkedIn time window parameter.
            timeout_seconds: Max network wait duration.
        """
        super().__init__(name="linkedin")
        settings = get_settings()
        self.user_data_dir = user_data_dir or settings.chrome_user_data_dir
        self.headless = headless
        self.time_filter = time_filter
        self.timeout_seconds = timeout_seconds

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
        for keyword in keywords:
            if yielded_count >= limit:
                break
            for loc in locations:
                if yielded_count >= limit:
                    break
                guest_failed = False
                extracted_for_pair = 0

                # 1. Tentative rapide via l'API publique de syndication
                try:
                    async for raw in self._scrape_guest_api(
                        keyword=keyword,
                        location=loc,
                        limit=limit - yielded_count,
                    ):
                        yield raw
                        yielded_count += 1
                        extracted_for_pair += 1
                        if yielded_count >= limit:
                            break

                except Exception as err:
                    print(
                        f"⚠️ [LinkedIn] Guest API restricted for '{keyword}' "
                        f"in '{loc}': {err}. Bascule vers Playwright..."
                    )
                    guest_failed = True

                # 2. Repli Playwright avec profil persistant si échec
                if (
                    guest_failed
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
                            if yielded_count >= limit:
                                break
                    except Exception as play_err:
                        print(
                            f"⚠️ [LinkedIn] Playwright fallback error for "
                            f"'{keyword}': {play_err}"
                        )

    async def _scrape_guest_api(
        self,
        keyword: str,
        location: str,
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Query LinkedIn guest syndication endpoint without browser overhead."""
        params: dict[str, str | int] = {
            "keywords": keyword,
            "location": location,
            "f_TPR": self.time_filter,
            "start": 0,
        }

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

            for block in card_blocks:
                title_match = re.search(
                    r'<h3 class="base-search-card__title">\s*(.*?)\s*</h3>',
                    block,
                    re.DOTALL,
                )
                company_match = re.search(
                    r'<h4 class="base-search-card__subtitle"[^>]*>\s*<a[^>]*>(.*?)</a>',
                    block,
                    re.DOTALL,
                )
                if not company_match:
                    company_match = re.search(
                        r'<h4 class="base-search-card__subtitle"[^>]*>\s*(.*?)\s*</h4>',
                        block,
                        re.DOTALL,
                    )
                loc_match = re.search(
                    r'<span class="job-search-card__location">\s*(.*?)\s*</span>',
                    block,
                    re.DOTALL,
                )
                link_match = re.search(
                    r'<a class="base-card__full-link"[^>]+'
                    r'href=["\'](https://[^"\']+)["\']',
                    block,
                )

                if not title_match or not link_match:
                    continue

                raw_title = _clean_text(title_match.group(1))
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
                canonical_url = f"https://www.linkedin.com/jobs/view/{job_id}"

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

    async def _fetch_guest_job_detail(
        self,
        client: httpx.AsyncClient,
        detail_url: str,
    ) -> tuple[str, bool, str | None]:
        """Fetch job description and identify if Easy Apply or external ATS."""
        try:
            res = await client.get(detail_url)
            if res.status_code != 200:
                return "", False, None

            text = res.text
            # Détection de la description
            desc_match = re.search(
                r'<div class="show-more-less-html__markup[^>]*>(.*?)</div>',
                text,
                re.DOTALL,
            )
            clean_desc = (
                _clean_text(desc_match.group(1)) if desc_match else ""
            )

            # Détection de Candidature Simplifiée (Easy Apply)
            is_easy = bool(
                "apply-button--easy" in text
                or "Candidature simplifiée" in text
                or "Easy Apply" in text
            )

            # Détection de lien externe vers ATS
            apply_url: str | None = None
            ext_match = re.search(
                r'href=["\'](https://[^"\']+applyUrl[^"\']+)["\']', text
            )
            if not ext_match:
                ext_match = re.search(
                    r'<a[^>]+class="[^"]*apply-button[^"]*"[^>]+'
                    r'href=["\'](https://[^"\']+)["\']',
                    text,
                )
            if ext_match:
                apply_url = ext_match.group(1)

            return clean_desc, is_easy, apply_url

        except Exception:
            return "", False, None

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
