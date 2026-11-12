"""Universal autonomous browser scraper for unhandled career portals and ATS."""

from collections.abc import AsyncIterator
import html
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import BrowserContext, Page, async_playwright

from config.settings import get_settings
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# Seuil minimal (caractères) avant navigation vers la fiche détaillée (réglable)
MIN_DOM_DESCRIPTION_CHARS: int = 400

# Sélecteurs fréquents de bannières de consentement cookies
COMMON_COOKIE_SELECTORS: list[str] = [
    "#onetrust-accept-btn-handler",
    "button[id*='cookie-accept']",
    "button[id*='accept-all']",
    "button[id*='didomi-notice-agree-button']",
    "button:has-text('Accepter')",
    "button:has-text('Accept all')",
    "button:has-text('J\'accepte')",
    "button:has-text('Agree')",
]


def _clean_text(raw_text: str | None) -> str:
    """Strip HTML tags, unescape entities, and condense whitespaces.

    Args:
        raw_text: Raw text string.

    Returns:
        str: Cleaned text.
    """
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    clean_text = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", clean_text).strip()


class GenericBrowserScraper(BaseScraper):
    """Universal Playwright scraper extracting jobs from any custom web portal.

    Leverages Chrome browser automation with persistent session profile,
    detecting Schema.org JobPosting JSON-LD metadata or extracting semantic
    DOM job cards as a fallback.

    Attributes:
        target_url: URL of target career page or job board.
        user_data_dir: Persistent Chrome user data directory.
        headless: Whether browser runs headless.
        timeout_seconds: Page navigation timeout in seconds.
    """

    def __init__(
        self,
        target_url: str,
        user_data_dir: Path | None = None,
        headless: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the generic browser scraper.

        Args:
            target_url: Target careers site URL.
            user_data_dir: Persistent browser profile directory.
            headless: Whether to run in headless mode.
            timeout_seconds: Max wait duration for page load.
        """
        # Nom de plateforme dérivé du domaine hôte
        parsed = urlparse(target_url)
        host_segment = parsed.netloc.replace(".", "_").lower()
        super().__init__(name=f"generic_{host_segment}")

        self.target_url = target_url
        self.headless = headless
        self.timeout_seconds = timeout_seconds

        settings = get_settings()
        self.user_data_dir = user_data_dir or settings.chrome_user_data_dir

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Navigate to target portal and extract published positions.

        Args:
            keywords: List of target job keywords.
            locations: Geographical criteria.
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Extracted job offer item.
        """
        profile_dir = self.user_data_dir
        if not profile_dir.exists():
            try:
                profile_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                profile_dir = Path("data/chrome_profile")
                profile_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            context: BrowserContext = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
                viewport={"width": 1280, "height": 800},
            )

            try:
                page: Page = await context.new_page()
                page.set_default_timeout(self.timeout_seconds * 1000)

                # Navigation vers l'URL cible
                try:
                    await page.goto(
                        self.target_url,
                        wait_until="domcontentloaded",
                    )
                except Exception as err:
                    print(
                        f"⚠️ [GenericScraper] Failed to load "
                        f"{self.target_url}: {err}"
                    )
                    return

                # Tentative d'acceptation des cookies
                await self._dismiss_cookie_banners(page)

                # Attente passive pour les applications SPA (React/Vue/Angular)
                await page.wait_for_timeout(2500)

                extracted_count = 0

                # 1. Tentative d'extraction prioritaire via Schema.org JSON-LD
                json_ld_jobs = await self._extract_json_ld_jobs(page)
                for job in json_ld_jobs:
                    if extracted_count >= limit:
                        break
                    yield job
                    extracted_count += 1

                # 2. Si aucune donnée JSON-LD n'est trouvée, extraction DOM
                if extracted_count == 0:
                    dom_jobs = await self._extract_dom_jobs(page, limit=limit)
                    for job in dom_jobs:
                        if extracted_count >= limit:
                            break
                        yield job
                        extracted_count += 1

            finally:
                await context.close()

    async def _dismiss_cookie_banners(self, page: Page) -> None:
        """Click on known cookie consent buttons if present."""
        for selector in COMMON_COOKIE_SELECTORS:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500):
                    await btn.click(timeout=1000)
                    break
            except Exception:
                continue

    async def _extract_json_ld_jobs(
        self, page: Page
    ) -> list[RawJobListing]:
        """Extract structured JobPosting listings from HTML script tags."""
        listings: list[RawJobListing] = []
        try:
            script_handles = await page.locator(
                'script[type="application/ld+json"]'
            ).all()

            for script in script_handles:
                text_content = await script.text_content()
                if not text_content:
                    continue

                try:
                    data = json.loads(text_content)
                except json.JSONDecodeError:
                    continue

                items_to_inspect: list[dict[str, Any]] = []
                if isinstance(data, dict):
                    if data.get("@type") == "JobPosting":
                        items_to_inspect.append(data)
                    elif data.get("@type") == "ItemList":
                        elements = data.get("itemListElement", [])
                        for el in elements:
                            if isinstance(el, dict):
                                item_obj = el.get("item", el)
                                if isinstance(item_obj, dict):
                                    items_to_inspect.append(item_obj)
                elif isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict) and obj.get("@type") == "JobPosting":
                            items_to_inspect.append(obj)

                for item in items_to_inspect:
                    title = _clean_text(item.get("title"))
                    if not title:
                        continue

                    # URL de l'offre
                    offer_url = item.get("url") or page.url
                    if not offer_url.startswith("http"):
                        offer_url = urljoin(page.url, offer_url)

                    # Nom de l'employeur
                    hiring_org = item.get("hiringOrganization", {})
                    company = (
                        hiring_org.get("name")
                        if isinstance(hiring_org, dict)
                        else "Entreprise"
                    )

                    # Localisation
                    loc_obj = item.get("jobLocation", {})
                    loc_str = ""
                    if isinstance(loc_obj, dict):
                        addr = loc_obj.get("address", {})
                        if isinstance(addr, dict):
                            loc_str = (
                                addr.get("addressLocality")
                                or addr.get("addressRegion")
                                or ""
                            )

                    desc = _clean_text(item.get("description") or title)
                    payload = dict(item)
                    payload.setdefault("is_easy_apply", False)
                    if item.get("employmentType") and "contract_type" not in payload:
                        payload["contract_type"] = str(item.get("employmentType"))

                    listings.append(
                        RawJobListing(
                            platform=self.name,
                            company=company or "Entreprise",
                            title=title,
                            url=offer_url,
                            description=desc,
                            location=loc_str,
                            raw_payload=payload,
                        )
                    )

        except Exception as err:
            print(f"⚠️ [GenericScraper] JSON-LD extraction error: {err}")

        return listings

    async def _fetch_detail_page(
        self,
        detail_page: Page,
        offer_url: str,
        fallback_company: str,
        fallback_title: str,
        fallback_text: str,
    ) -> RawJobListing:
        """Visit a DOM card's target URL to extract full job description.

        Args:
            detail_page: Reusable Playwright detail page.
            offer_url: Target offer URL.
            fallback_company: Company inferred from listing page.
            fallback_title: Title inferred from DOM card.
            fallback_text: Short card text fallback.

        Returns:
            RawJobListing: Enriched listing if detail text was found.
        """
        payload: dict[str, Any] = {
            "scraped_text": fallback_text,
            "is_easy_apply": False,
        }
        try:
            await detail_page.goto(offer_url, wait_until="domcontentloaded")
            await detail_page.wait_for_timeout(1200)

            json_ld_jobs = await self._extract_json_ld_jobs(detail_page)
            if json_ld_jobs:
                best = json_ld_jobs[0]
                return RawJobListing(
                    platform=self.name,
                    company=best.company or fallback_company,
                    title=best.title or fallback_title,
                    url=offer_url,
                    description=best.description or fallback_text,
                    location=best.location,
                    raw_payload=best.raw_payload,
                )

            main_loc = detail_page.locator("main, article, [role='main']").first
            if await main_loc.count() > 0:
                body_text = _clean_text(await main_loc.text_content())
            else:
                body_text = _clean_text(
                    await detail_page.locator("body").text_content()
                )
            if len(body_text) > len(fallback_text):
                return RawJobListing(
                    platform=self.name,
                    company=fallback_company,
                    title=fallback_title,
                    url=offer_url,
                    description=body_text[:8000],
                    location="",
                    raw_payload=payload,
                )
        except Exception:
            pass

        return RawJobListing(
            platform=self.name,
            company=fallback_company,
            title=fallback_title,
            url=offer_url,
            description=fallback_text,
            location="",
            raw_payload=payload,
        )

    async def _extract_dom_jobs(
        self,
        page: Page,
        limit: int = 20,
    ) -> list[RawJobListing]:
        """Heuristic DOM parsing extracting job cards and career links."""
        listings: list[RawJobListing] = []
        candidates: list[tuple[str, str, str, str]] = []
        seen_urls: set[str] = set()
        try:
            # Sélecteurs courants de cartes ou liens d'offres
            card_selectors = [
                "article",
                "[class*='job-card']",
                "[class*='job-item']",
                "[class*='offer-card']",
                "li[class*='job']",
                "div[data-job-id]",
                "a[href*='/job/']",
                "a[href*='/career/']",
                "a[href*='/offre/']",
            ]

            cards = await page.locator(", ".join(card_selectors)).all()

            # Extraction du nom de l'entreprise via le titre de la page
            page_title = await page.title()
            default_company = (
                page_title.split("-")[0].split("|")[0].strip() or "Entreprise"
            )

            for card in cards:
                if len(candidates) >= limit:
                    break

                # Recherche de titre et lien
                raw_content = await card.text_content()
                text = _clean_text(raw_content)
                if not text or len(text) < 10:
                    continue

                # Recherche d'un lien
                tag_name = await card.evaluate("el => el.tagName.toLowerCase()")
                link_href: str | None = None
                if tag_name == "a":
                    link_href = await card.get_attribute("href")
                else:
                    first_link = card.locator("a").first
                    if await first_link.count() > 0:
                        link_href = await first_link.get_attribute("href")

                if not link_href:
                    continue

                full_url = urljoin(page.url, link_href)
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                raw_lines = [
                    line.strip()
                    for line in (raw_content or "").split("\n")
                    if line.strip()
                ]
                job_title = (raw_lines[0] if raw_lines else text)[:120]
                if is_excluded_title(job_title):
                    continue

                candidates.append(
                    (default_company, job_title, full_url, text)
                )

            if candidates:
                detail_page = await page.context.new_page()
                detail_page.set_default_timeout(self.timeout_seconds * 1000)
                try:
                    for company, job_title, full_url, card_text in candidates:
                        if (
                            len(card_text) < MIN_DOM_DESCRIPTION_CHARS
                            and full_url != page.url
                        ):
                            enriched = await self._fetch_detail_page(
                                detail_page=detail_page,
                                offer_url=full_url,
                                fallback_company=company,
                                fallback_title=job_title,
                                fallback_text=card_text,
                            )
                            if not is_excluded_title(enriched.title):
                                listings.append(enriched)
                        else:
                            listings.append(
                                RawJobListing(
                                    platform=self.name,
                                    company=company,
                                    title=job_title,
                                    url=full_url,
                                    description=card_text,
                                    location="",
                                    raw_payload={
                                        "scraped_text": card_text,
                                        "is_easy_apply": False,
                                    },
                                )
                            )
                finally:
                    await detail_page.close()

        except Exception as err:
            print(f"⚠️ [GenericScraper] DOM extraction error: {err}")

        return listings
