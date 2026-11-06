"""Browser automation module powered by Playwright with persistent sessions."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field

from config.settings import get_settings


class CandidateDetails(BaseModel):
    """Personal contact and profile details used for form auto-filling.

    Attributes:
        first_name: Candidate given name.
        last_name: Candidate family name.
        email: Candidate contact email address.
        phone: Phone number formatted with country code.
        phone_national: Phone number formatted for domestic forms.
        city: Resident city.
        postal_code: Resident postal code.
        linkedin_url: Public LinkedIn profile URL.
        github_url: Public GitHub profile URL.
    """

    first_name: str = Field(default="Jérémy", description="Given name.")
    last_name: str = Field(default="Foeller", description="Family name.")
    email: str = Field(
        default="jeremy.foeller@gmail.com",
        description="Contact email.",
    )
    phone: str = Field(
        default="+33 6 24 05 81 85",
        description="Phone with country code.",
    )
    phone_national: str = Field(
        default="0624058185",
        description="National phone digits.",
    )
    city: str = Field(default="Allauch", description="City of residence.")
    postal_code: str = Field(
        default="13190",
        description="Postal code.",
    )
    linkedin_url: str = Field(
        default="https://www.linkedin.com/in/jérémy-foeller-95207a70",
        description="LinkedIn profile URL.",
    )
    github_url: str = Field(
        default="https://github.com/jefo1383",
        description="GitHub profile URL.",
    )


class BrowserRunner:
    """Manages persistent browser sessions for job application automation.

    Attributes:
        user_data_dir: Filesystem path to the persistent Chrome user profile.
        resume_pdf_path: Filesystem path to the candidate's PDF resume.
        headless: Whether to execute browser without graphical display.
        slow_mo: Milliseconds to wait between actions for humanized typing.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        resume_pdf_path: Path | None = None,
        headless: bool = False,
        slow_mo: int = 300,
    ) -> None:
        """Initialize the browser automation runner.

        Args:
            user_data_dir: Custom Chrome user data directory.
            resume_pdf_path: Custom PDF resume file path.
            headless: Display mode (default False for user supervision).
            slow_mo: Milliseconds delay between browser actions.
        """
        settings = get_settings()

        self.user_data_dir: Path = (
            user_data_dir or settings.chrome_user_data_dir
        )
        self.resume_pdf_path: Path = (
            resume_pdf_path or settings.resume_pdf_path
        )
        self.headless: bool = headless
        self.slow_mo: int = slow_mo

        # Création du dossier de profil persistant avec repli sécurisé
        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.user_data_dir = (
                settings.database_path.parent / "chrome_profile"
            ).resolve()
            self.user_data_dir.mkdir(parents=True, exist_ok=True)

    async def _fill_common_fields(
        self,
        page: Page,
        candidate: CandidateDetails,
        cover_letter: str | None = None,
    ) -> int:
        """Locate and fill standard contact input fields across ATS forms.

        Args:
            page: Active Playwright page.
            candidate: Candidate details to inject.
            cover_letter: Optional cover letter text.

        Returns:
            int: Number of fields successfully populated.
        """
        filled_count = 0

        # Mapping des sélecteurs CSS courants pour les formulaires ATS
        field_mappings = [
            # Email
            (
                "input[type='email'], input[name*='email' i], "
                "input[id*='email' i]",
                candidate.email,
            ),
            # Prénom
            (
                "input[name*='firstname' i], input[name*='first_name' i], "
                "input[name*='prenom' i], input[id*='firstname' i]",
                candidate.first_name,
            ),
            # Nom
            (
                "input[name*='lastname' i], input[name*='last_name' i], "
                "input[name*='nom' i], input[id*='lastname' i]",
                candidate.last_name,
            ),
            # Téléphone
            (
                "input[type='tel'], input[name*='phone' i], "
                "input[name*='telephone' i], input[id*='phone' i]",
                candidate.phone,
            ),
            # Ville
            (
                "input[name*='city' i], input[name*='ville' i], "
                "input[id*='city' i]",
                candidate.city,
            ),
            # Code postal
            (
                "input[name*='zip' i], input[name*='postal' i], "
                "input[name*='code_postal' i]",
                candidate.postal_code,
            ),
            # LinkedIn
            (
                "input[name*='linkedin' i], input[id*='linkedin' i]",
                candidate.linkedin_url,
            ),
        ]

        for selector, value in field_mappings:
            try:
                locator = page.locator(selector).first
                if await locator.is_visible(timeout=1000):
                    # Remplissage si le champ est actuellement vide
                    current_val = await locator.input_value()
                    if not current_val:
                        await locator.fill(value)
                        filled_count += 1
            except Exception:
                continue

        # Téléversement automatique du CV PDF si un champ d'upload existe
        if self.resume_pdf_path.exists():
            try:
                file_input = page.locator("input[type='file']").first
                if await file_input.count() > 0:
                    await file_input.set_input_files(str(self.resume_pdf_path))
                    filled_count += 1
                    print(f"📄 CV téléversé : {self.resume_pdf_path.name}")
            except Exception as err:
                print(f"⚠️ Échec du téléversement automatique du CV : {err}")

        # Remplissage de la lettre de motivation si demandée
        if cover_letter:
            try:
                letter_area = page.locator(
                    "textarea[name*='cover' i], textarea[name*='lettre' i], "
                    "textarea[name*='motivation' i], textarea[id*='letter' i]"
                ).first
                if await letter_area.is_visible(timeout=1000):
                    await letter_area.fill(cover_letter)
                    filled_count += 1
            except Exception:
                pass

        return filled_count

    async def open_and_prefill_job(
        self,
        url: str,
        candidate: CandidateDetails | None = None,
        cover_letter: str | None = None,
        keep_open_seconds: int = 120,
    ) -> bool:
        """Open ATS job application in visible browser, prefill and wait for user.

        Args:
            url: Job offer application URL.
            candidate: Candidate profile details (defaults to standard).
            cover_letter: Tailored cover letter to populate if requested.
            keep_open_seconds: Time to keep window active for manual submit.

        Returns:
            bool: True if page loaded and prefilled successfully.
        """
        details = candidate or CandidateDetails()
        print(f"🌐 [Playwright] Ouverture du portail ATS : {url}")

        async with async_playwright() as p:
            # Lancement du contexte persistant réutilisant les cookies
            context: BrowserContext = await p.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=["--start-maximized"],
            )

            try:
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)

                # Tentative d'auto-remplissage des champs standards
                filled = await self._fill_common_fields(
                    page=page,
                    candidate=details,
                    cover_letter=cover_letter,
                )
                print(
                    f"✓ [Playwright] {filled} champ(s) pré-rempli(s). "
                    f"Fenêtre active pendant {keep_open_seconds}s "
                    "pour validation manuelle."
                )

                # Attente pour permettre la validation finale manuelle par l'utilisateur
                await asyncio.sleep(keep_open_seconds)
                return True

            except Exception as err:
                print(f"⚠️ [Playwright] Erreur lors de la navigation : {err}")
                return False
            finally:
                await context.close()

    async def apply_linkedin_easy_apply(
        self,
        url: str,
        candidate: CandidateDetails | None = None,
        screenshot_dir: Path | None = None,
    ) -> Path | None:
        """Handle LinkedIn Easy Apply flow and capture confirmation screenshot.

        Args:
            url: Target LinkedIn job view URL.
            candidate: Candidate details for contact forms.
            screenshot_dir: Destination folder for confirmation capture.

        Returns:
            Path | None: Path to screenshot if process completed, None otherwise.
        """
        details = candidate or CandidateDetails()
        output_dir = screenshot_dir or (Path("data") / "screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        screenshot_path = output_dir / f"easy_apply_{timestamp}.png"

        print(f"💼 [Playwright] Démarrage du flux Easy Apply : {url}")

        async with async_playwright() as p:
            context: BrowserContext = await p.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                slow_mo=self.slow_mo,
            )

            try:
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)

                # Détection du bouton 'Candidature simplifiée' ou 'Easy Apply'
                apply_button = page.locator(
                    "button.jobs-apply-button, "
                    "button:has-text('Candidature simplifiée'), "
                    "button:has-text('Easy Apply')"
                ).first

                if not await apply_button.is_visible(timeout=5000):
                    print("⚠️ Bouton 'Easy Apply' non détecté sur la page.")
                    # Capture d'écran du blocage éventuel
                    await page.screenshot(path=str(screenshot_path))
                    return screenshot_path

                # Clic pour ouvrir la modale LinkedIn
                await apply_button.click()
                await page.wait_for_timeout(1500)

                # Remplissage des formulaires de la modale
                await self._fill_common_fields(page, details)

                # Capture d'écran d'étape / confirmation
                await page.screenshot(path=str(screenshot_path))
                print(f"📸 Capture d'écran enregistrée : {screenshot_path}")
                return screenshot_path

            except Exception as err:
                print(f"⚠️ [Playwright] Erreur flux Easy Apply : {err}")
                return None
            finally:
                await context.close()
