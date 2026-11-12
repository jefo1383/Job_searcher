"""Scraper for Indeed supporting headless cycle and interactive browser flow."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from urllib.parse import quote_plus

from playwright.async_api import BrowserContext, Page, async_playwright

from config.settings import get_settings
from core.location_matcher import LocationMatcher, normalize_geo_text
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# URL racine de recherche Indeed France
INDEED_BASE_SEARCH_URL: str = "https://fr.indeed.com/emplois"

# Signature type pour le callback d'attente utilisateur en mode interactif
ReadyCallback = Callable[[str], Awaitable[bool]]


class IndeedScraper(BaseScraper):
    """Asynchronous scraper extracting job postings from Indeed via Playwright.

    Attributes:
        user_data_dir: Persistent Chrome profile directory to reuse sessions.
        radius: Kilometers radius around the target location.
        fromage: Maximum age of postings in days (e.g. 7 or 14).
        headless: Whether to execute browser in headless mode.
        interactive: If True, opens visible browser and awaits user confirmation.
        wait_for_ready: Optional async callback prompting user once page opens.
        scroll_steps: Number of incremental scrolls per results page.
        scroll_delay_ms: Wait delay between incremental scroll steps in ms.
        location_matcher: Shared geo matcher enforcing 25km radius in dept 13.
        last_run_blocked: True if the last scrape stopped due to anti-bot block.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        radius: int = 25,
        fromage: int = 14,
        headless: bool = True,
        interactive: bool = False,
        wait_for_ready: ReadyCallback | None = None,
        scroll_steps: int = 4,
        scroll_delay_ms: int = 600,
        location_matcher: LocationMatcher | None = None,
    ) -> None:
        """Initialize the Indeed scraper.

        Args:
            user_data_dir: Chrome user data directory (defaults to settings).
            radius: Geographical radius search filter in km.
            fromage: Maximum posting age in days.
            headless: Display mode for browser navigation.
            interactive: Wait for manual user browser prep before scrolling.
            wait_for_ready: Async callback receiving URL and returning proceed.
            scroll_steps: Incremental scroll count to reveal lazy job cards.
            scroll_delay_ms: Pause between scroll steps in milliseconds.
            location_matcher: Shared geo matcher for dept 13 filtering.
        """
        super().__init__(name="indeed")
        settings = get_settings()

        self.user_data_dir: Path = (
            user_data_dir or settings.chrome_user_data_dir
        )
        self.radius = radius
        self.fromage = fromage
        self.headless = False if interactive else headless
        self.interactive = interactive
        self.wait_for_ready = wait_for_ready
        self.scroll_steps = scroll_steps
        self.scroll_delay_ms = scroll_delay_ms
        self.location_matcher = location_matcher or LocationMatcher()
        self.last_run_blocked: bool = False

        # Création sécurisée du dossier de profil avec repli local si restreint
        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.user_data_dir = (
                settings.database_path.parent / "chrome_profile"
            ).resolve()
            self.user_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_search_location(locations: list[str]) -> str:
        """Select a human-readable location query from configured targets.

        Args:
            locations: List of target geographical filters.

        Returns:
            str: City or department name compatible with Indeed search.
        """
        for loc in locations:
            cleaned = loc.strip()
            if cleaned and not cleaned.isdigit():
                return cleaned
        return "Bouches-du-Rhône"

    @staticmethod
    async def _default_terminal_wait(target_url: str) -> bool:
        """Prompt user in terminal until Indeed is manually loaded in Chrome."""
        print(
            "\n🌐 [Indeed Interactive] Fenêtre Chrome ouverte sur :\n"
            f"   {target_url}\n"
            "👉 Ouvrez/validez Indeed dans le navigateur si nécessaire, puis "
            "appuyez sur [Entrée] pour lancer le scroll (ou tapez 'q' + "
            "Entrée pour annuler) :"
        )
        answer = await asyncio.to_thread(input, "")
        return answer.strip().lower() not in ("q", "quit", "cancel", "non")

    @staticmethod
    async def _is_page_blocked(page: Page, status_code: int | None) -> bool:
        """Detect whether Indeed presented an anti-bot or rate-limit screen."""
        if status_code in (403, 429):
            return True
        try:
            page_title = (await page.title()).lower()
        except Exception:
            page_title = ""
        return any(
            tok in page_title
            for tok in (
                "blocked",
                "cloudflare",
                "just a moment",
                "captcha",
                "attention required",
            )
        )

    @staticmethod
    async def _dismiss_cookies(page: Page) -> None:
        """Dismiss Indeed cookie consent dialog politely if shown."""
        selectors = (
            "button#onetrust-reject-all-handler",
            "button:has-text('Tout refuser')",
            "button#onetrust-accept-btn-handler",
            "button:has-text('Accepter')",
        )
        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=800):
                    await btn.click()
                    return
            except Exception:
                continue

    async def _scroll_results_page(self, page: Page) -> None:
        """Scroll incrementally through the page so lazy cards render cleanly."""
        for _ in range(max(1, self.scroll_steps)):
            try:
                await page.evaluate("window.scrollBy(0, 750)")
                await page.wait_for_timeout(self.scroll_delay_ms)
            except Exception:
                break
        try:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(300)
        except Exception:
            pass

    @staticmethod
    def _is_remote_hint(title: str, location: str) -> bool:
        """Return True when title or location indicates full remote work."""
        norm = f"{normalize_geo_text(title)} {normalize_geo_text(location)}"
        return any(
            tok in norm
            for tok in ("remote", "teletravail", "full remote", "distanciel")
        )

    async def _extract_cards_from_page(
        self,
        page: Page,
        keyword: str,
        search_loc: str,
        allowed_locations: list[str],
        seen_jks: set[str],
        remaining_limit: int,
    ) -> list[RawJobListing]:
        """Scroll and extract full job listings from the currently loaded page.

        Args:
            page: Active Playwright Page containing Indeed job cards.
            keyword: Active keyword label for metadata tracking.
            search_loc: Fallback city/area label.
            allowed_locations: Target criteria locations for geo validation.
            seen_jks: Mutable set tracking already collected Indeed job keys.
            remaining_limit: Maximum jobs left to collect in this cycle.

        Returns:
            list[RawJobListing]: Extracted listings from the current page.
        """
        results: list[RawJobListing] = []
        if remaining_limit <= 0:
            return results

        await self._dismiss_cookies(page)

        card_locator = page.locator("div.job_seen_beacon, div.cardOutline")
        try:
            await card_locator.first.wait_for(timeout=6000)
        except Exception:
            return results

        await self._scroll_results_page(page)
        count = await card_locator.count()

        for idx in range(count):
            if len(results) >= remaining_limit:
                break

            card = card_locator.nth(idx)
            try:
                await card.scroll_into_view_if_needed(timeout=2500)
            except Exception:
                pass

            title_link = card.locator("a.jcs-JobTitle, a[data-jk]").first
            if await title_link.count() == 0:
                continue

            job_title = (await title_link.inner_text()).strip()
            jk = await title_link.get_attribute("data-jk")
            if not jk:
                jk = await card.get_attribute("data-jk")
            if not jk:
                href = await title_link.get_attribute("href")
                if href and "jk=" in href:
                    jk_parts = [
                        part.split("=")[1]
                        for part in href.split("&")
                        if "jk=" in part
                    ]
                    if jk_parts:
                        jk = jk_parts[0]

            if not jk or not job_title or jk in seen_jks:
                continue
            if is_excluded_title(job_title):
                continue

            company_name = "Entreprise sur Indeed"
            company_loc = card.locator(
                "[data-testid='company-name'], span.css-63koeb"
            ).first
            if await company_loc.count() > 0:
                company_name = (await company_loc.inner_text()).strip()

            location_txt = search_loc
            location_loc = card.locator(
                "[data-testid='text-location'], div.css-1p0sjhy"
            ).first
            if await location_loc.count() > 0:
                location_txt = (await location_loc.inner_text()).strip()

            if location_txt and not self._is_remote_hint(
                job_title, location_txt
            ):
                allowed = await self.location_matcher.is_location_allowed(
                    raw_location=location_txt,
                    extra_locations=allowed_locations,
                )
                if not allowed:
                    continue

            seen_jks.add(jk)
            canonical_url = f"https://fr.indeed.com/viewjob?jk={jk}"

            snippet_txt = ""
            snippet_loc = card.locator(
                "div.job-snippet, ul.css-ky7n0f, "
                "div[data-testid='job-snippet']"
            ).first
            if await snippet_loc.count() > 0:
                snippet_txt = (await snippet_loc.inner_text()).strip()

            # Clic sur la carte pour lire le panneau de description complète
            full_pane_text = ""
            try:
                await title_link.click(timeout=3000)
                desc_pane = page.locator("#jobDescriptionText").first
                await desc_pane.wait_for(state="visible", timeout=3500)
                full_pane_text = (await desc_pane.inner_text()).strip()
            except Exception:
                full_pane_text = ""

            body_text = full_pane_text or snippet_txt or job_title
            full_description = (
                f"{job_title} chez {company_name}\n"
                f"Lieu : {location_txt}\n\n"
                f"Description de l'offre :\n{body_text}"
            )

            results.append(
                RawJobListing(
                    platform=self.name,
                    company=company_name,
                    title=job_title,
                    url=canonical_url,
                    description=full_description,
                    location=location_txt,
                    raw_payload={
                        "jk": jk,
                        "keyword": keyword,
                        "is_easy_apply": False,
                        "apply_url": canonical_url,
                    },
                )
            )
            await asyncio.sleep(0.3)

        return results

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Scrape Indeed job listings in automatic or interactive mode.

        Args:
            keywords: List of target job keywords.
            locations: Geographical filters (default Bouches-du-Rhône).
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Normalized job post items.
        """
        self.last_run_blocked = False
        search_loc = self._resolve_search_location(locations)
        yielded_count = 0
        seen_jks: set[str] = set()
        effective_headless = False if self.interactive else self.headless

        async with async_playwright() as p:
            try:
                context: BrowserContext = (
                    await p.chromium.launch_persistent_context(
                        user_data_dir=str(self.user_data_dir),
                        headless=effective_headless,
                        slow_mo=200,
                        viewport={"width": 1280, "height": 850},
                        args=[
                            "--disable-blink-features=AutomationControlled",
                            "--no-sandbox",
                        ],
                    )
                )
            except Exception as err:
                print(
                    f"⚠️ [IndeedScraper] Échec du lancement de Playwright: {err}"
                )
                return

            try:
                page = (
                    context.pages[0]
                    if context.pages
                    else await context.new_page()
                )

                # En mode interactif, on ouvre d'abord la fenêtre à l'utilisateur
                if self.interactive:
                    first_kw = keywords[0] if keywords else "Développeur"
                    start_url = (
                        f"{INDEED_BASE_SEARCH_URL}?"
                        f"q={quote_plus(first_kw)}&"
                        f"l={quote_plus(search_loc)}&"
                        f"radius={self.radius}&"
                        f"fromage={self.fromage}"
                    )
                    try:
                        await page.goto(
                            start_url,
                            wait_until="domcontentloaded",
                            timeout=25000,
                        )
                    except Exception:
                        pass

                    ready_cb = self.wait_for_ready or self._default_terminal_wait
                    proceed = await ready_cb(page.url)
                    if not proceed:
                        print("ℹ️ [IndeedScraper] Session interactive annulée.")
                        return

                    # Extraction prioritaire de la page déjà ouverte par l'humain
                    if not await self._is_page_blocked(page, None):
                        current_items = await self._extract_cards_from_page(
                            page=page,
                            keyword="manual_open_tab",
                            search_loc=search_loc,
                            allowed_locations=locations,
                            seen_jks=seen_jks,
                            remaining_limit=limit - yielded_count,
                        )
                        for item in current_items:
                            yield item
                            yielded_count += 1
                            if yielded_count >= limit:
                                return

                # Parcours des mots-clés configurés dans search_criteria.txt
                for idx, keyword in enumerate(keywords):
                    if yielded_count >= limit:
                        break
                    # En interactif, éviter de recharger immédiatement l'onglet 0
                    if self.interactive and idx == 0 and yielded_count > 0:
                        continue

                    query_url = (
                        f"{INDEED_BASE_SEARCH_URL}?"
                        f"q={quote_plus(keyword)}&"
                        f"l={quote_plus(search_loc)}&"
                        f"radius={self.radius}&"
                        f"fromage={self.fromage}"
                    )

                    try:
                        response = await page.goto(
                            query_url,
                            wait_until="domcontentloaded",
                            timeout=25000,
                        )
                        status = response.status if response else None
                        if await self._is_page_blocked(page, status):
                            self.last_run_blocked = True
                            status_txt = status if status is not None else "N/A"
                            print(
                                "⚠️ [IndeedScraper] Accès bloqué par Indeed "
                                f"(HTTP {status_txt}). Utilisez /indeed ou "
                                "--indeed pour ouvrir le navigateur manuellement."
                            )
                            break

                        items = await self._extract_cards_from_page(
                            page=page,
                            keyword=keyword,
                            search_loc=search_loc,
                            allowed_locations=locations,
                            seen_jks=seen_jks,
                            remaining_limit=limit - yielded_count,
                        )
                        for item in items:
                            yield item
                            yielded_count += 1
                            if yielded_count >= limit:
                                break

                    except Exception as err:
                        print(
                            f"⚠️ [IndeedScraper] Erreur sur '{keyword}': {err}"
                        )
                        continue

            finally:
                await context.close()
