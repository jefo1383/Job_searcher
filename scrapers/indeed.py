"""Scraper implementation for Indeed using Playwright with persistent sessions."""

from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote_plus
from playwright.async_api import BrowserContext, async_playwright

from config.settings import get_settings
from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# URL racine de recherche Indeed France
INDEED_BASE_SEARCH_URL: str = "https://fr.indeed.com/emplois"


class IndeedScraper(BaseScraper):
    """Asynchronous scraper extracting job postings from Indeed via Playwright.

    Attributes:
        user_data_dir: Persistent Chrome profile directory to reuse sessions.
        radius: Kilometers radius around the target location.
        fromage: Maximum age of postings in days (e.g. 7 or 14).
        headless: Whether to execute browser in headless mode.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        radius: int = 25,
        fromage: int = 14,
        headless: bool = True,
    ) -> None:
        """Initialize the Indeed scraper.

        Args:
            user_data_dir: Chrome user data directory (defaults to settings).
            radius: Geographical radius search filter in km.
            fromage: Maximum posting age in days.
            headless: Display mode for browser navigation.
        """
        super().__init__(name="indeed")
        settings = get_settings()

        self.user_data_dir: Path = (
            user_data_dir or settings.chrome_user_data_dir
        )
        self.radius = radius
        self.fromage = fromage
        self.headless = headless

        # Création sécurisée du dossier de profil avec repli local si restreint
        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.user_data_dir = (
                settings.database_path.parent / "chrome_profile"
            ).resolve()
            self.user_data_dir.mkdir(parents=True, exist_ok=True)

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Scrape Indeed job listings matching target queries using Playwright.

        Args:
            keywords: List of target job keywords.
            locations: Geographical filters (default Bouches-du-Rhône).
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Normalized job post items.
        """
        search_loc = locations[0] if locations else "Bouches-du-Rhône"
        yielded_count = 0

        async with async_playwright() as p:
            try:
                # Contexte persistant pour contourner les protections anti-bot
                context: BrowserContext = (
                    await p.chromium.launch_persistent_context(
                        user_data_dir=str(self.user_data_dir),
                        headless=self.headless,
                        slow_mo=200,
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
                page = await context.new_page()

                for keyword in keywords:
                    if yielded_count >= limit:
                        break

                    query_url = (
                        f"{INDEED_BASE_SEARCH_URL}?"
                        f"q={quote_plus(keyword)}&"
                        f"l={quote_plus(search_loc)}&"
                        f"radius={self.radius}&"
                        f"fromage={self.fromage}"
                    )

                    try:
                        # Chargement de la page de résultats
                        await page.goto(
                            query_url,
                            wait_until="domcontentloaded",
                            timeout=30000,
                        )

                        # Fermeture éventuelle de la modale de cookies
                        try:
                            cookie_btn = page.locator(
                                "button#onetrust-accept-btn-handler, "
                                "button:has-text('Accepter')"
                            ).first
                            if await cookie_btn.is_visible(timeout=1500):
                                await cookie_btn.click()
                        except Exception:
                            pass

                        # Attente de la présence des cartes d'offres
                        card_locator = page.locator(
                            "div.job_seen_beacon, div.cardOutline"
                        )
                        await card_locator.first.wait_for(timeout=7000)

                        count = await card_locator.count()
                        if count == 0:
                            continue

                        for idx in range(count):
                            if yielded_count >= limit:
                                break

                            card = card_locator.nth(idx)

                            # Détection du lien et du JobKey (jk)
                            title_link = card.locator(
                                "a.jcs-JobTitle, a[data-jk]"
                            ).first
                            if await title_link.count() == 0:
                                continue

                            job_title = (
                                await title_link.inner_text()
                            ).strip()
                            jk = await title_link.get_attribute("data-jk")

                            if not jk:
                                # Tentative de lecture du jk sur la carte parente
                                jk = await card.get_attribute("data-jk")

                            if not jk:
                                href = await title_link.get_attribute("href")
                                if href and "jk=" in href:
                                    jk_match = [
                                        part.split("=")[1]
                                        for part in href.split("&")
                                        if "jk=" in part
                                    ]
                                    if jk_match:
                                        jk = jk_match[0]

                            if not jk or not job_title:
                                continue

                            # URL canonique dédupliquée pour Indeed
                            canonical_url = f"https://fr.indeed.com/viewjob?jk={jk}"

                            # Nom de l'entreprise
                            company_name = "Entreprise sur Indeed"
                            company_loc = card.locator(
                                "[data-testid='company-name'], span.css-63koeb"
                            ).first
                            if await company_loc.count() > 0:
                                company_name = (
                                    await company_loc.inner_text()
                                ).strip()

                            # Localisation du poste
                            location_txt = search_loc
                            location_loc = card.locator(
                                "[data-testid='text-location'], div.css-1p0sjhy"
                            ).first
                            if await location_loc.count() > 0:
                                location_txt = (
                                    await location_loc.inner_text()
                                ).strip()

                            # Extrait de description
                            snippet_txt = ""
                            snippet_loc = card.locator(
                                "div.job-snippet, ul.css-ky7n0f, "
                                "div[data-testid='job-snippet']"
                            ).first
                            if await snippet_loc.count() > 0:
                                snippet_txt = (
                                    await snippet_loc.inner_text()
                                ).strip()

                            full_description = (
                                f"{job_title} chez {company_name}\n"
                                f"Lieu : {location_txt}\n\n"
                                f"Résumé de l'offre :\n{snippet_txt}"
                            )

                            yield RawJobListing(
                                platform=self.name,
                                company=company_name,
                                title=job_title,
                                url=canonical_url,
                                description=full_description,
                                location=location_txt,
                                raw_payload={"jk": jk, "keyword": keyword},
                            )
                            yielded_count += 1

                    except Exception as err:
                        print(
                            f"⚠️ [IndeedScraper] Erreur lors de la requête "
                            f"'{keyword}': {err}"
                        )
                        continue

            finally:
                await context.close()
