"""Browser automation module powered by Playwright with persistent sessions."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import (
    BrowserContext,
    Locator,
    Page,
    async_playwright,
)
from pydantic import BaseModel, Field

from config.settings import get_settings
from core.profile_builder import find_latest_resume
from core.question_solver import ScreeningQuestionSolver


class ApplicationModalResult(BaseModel):
    """Structured outcome returned by multi-step browser application flow.

    Attributes:
        status: Terminal outcome code ('ready_for_manual_submit',
            'blocked_on_validation', 'apply_button_not_found', or 'error').
        screenshot_path: Saved confirmation screenshot path if available.
        steps_completed: Number of modal steps traversed.
        fields_filled: Total input/select/radio fields populated.
        unanswered_questions: Labels of questions left empty for user review.
    """

    status: str = Field(..., description="Outcome status code.")
    screenshot_path: Path | None = Field(
        default=None,
        description="Confirmation or blocker screenshot path.",
    )
    steps_completed: int = Field(
        default=0,
        description="Number of modal steps processed.",
    )
    fields_filled: int = Field(
        default=0,
        description="Number of form fields auto-filled.",
    )
    unanswered_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring manual input by the user.",
    )


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
        question_solver: Hybrid solver handling unknown screening questions.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        resume_pdf_path: Path | None = None,
        headless: bool = False,
        slow_mo: int = 300,
        question_solver: ScreeningQuestionSolver | None = None,
    ) -> None:
        """Initialize the browser automation runner.

        Args:
            user_data_dir: Custom Chrome user data directory.
            resume_pdf_path: Custom PDF resume file path.
            headless: Display mode (default False for user supervision).
            slow_mo: Milliseconds delay between browser actions.
            question_solver: Optional custom screening question solver.
        """
        settings = get_settings()

        self.user_data_dir: Path = (
            user_data_dir or settings.chrome_user_data_dir
        )
        self.resume_pdf_path: Path = (
            resume_pdf_path or settings.resume_pdf_path
        )
        self._profile_dir: Path = settings.profile_dir
        self.headless: bool = headless
        self.slow_mo: int = slow_mo
        self.question_solver: ScreeningQuestionSolver = (
            question_solver or ScreeningQuestionSolver()
        )

        # Création du dossier de profil persistant avec repli sécurisé
        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.user_data_dir = (
                settings.database_path.parent / "chrome_profile"
            ).resolve()
            self.user_data_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_resume_path(self) -> Path | None:
        """Return active resume path, falling back to latest file in Profile/.

        Returns:
            Path | None: Existing resume path or None if unavailable.
        """
        if self.resume_pdf_path.exists():
            return self.resume_pdf_path
        return find_latest_resume(self._profile_dir)

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

        # Téléversement automatique du CV PDF si un champ d'upload vide existe
        active_resume = self._resolve_resume_path()
        if active_resume is not None and active_resume.exists():
            try:
                file_input = page.locator("input[type='file']").first
                if await file_input.count() > 0:
                    current_file = await file_input.input_value()
                    if not current_file:
                        await file_input.set_input_files(str(active_resume))
                        filled_count += 1
                        print(f"📄 CV téléversé : {active_resume.name}")
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

    async def _handle_cookie_banner(self, page: Page) -> str:
        """Dismiss cookie consent banners by refusing first, accepting as fallback.

        Args:
            page: Active Playwright page.

        Returns:
            str: 'rejected', 'accepted', or 'none' if no banner was detected.
        """
        reject_selectors = (
            "button#onetrust-reject-all-handler",
            "button#didomi-notice-disagree-button",
            "button:has-text('Tout refuser')",
            "button:has-text('Refuser tout')",
            "button:has-text('Continuer sans accepter')",
            "a:has-text('Continuer sans accepter')",
            "button:has-text('Reject all')",
            "button:has-text('Decline all')",
            "button:has-text('Refuser')",
            "button:has-text('Decline')",
            "button:has-text('Reject')",
            "button[data-testid*='reject' i]",
        )
        accept_selectors = (
            "button#onetrust-accept-btn-handler",
            "button#didomi-notice-agree-button",
            "button:has-text('Tout accepter')",
            "button:has-text('Accepter tout')",
            "button:has-text(\"J'accepte\")",
            "button:has-text('Accept all')",
            "button:has-text('Accepter')",
            "button:has-text('Accept')",
            "button[data-testid*='accept' i]",
        )

        # Recherche sur la page principale puis dans d'éventuelles iframes CMP
        scopes = [page.main_frame, *[f for f in page.frames if f != page.main_frame]]
        for action_name, selectors, log_msg in (
            ("rejected", reject_selectors, "🍪 Bandeau cookies refusé par défaut."),
            (
                "accepted",
                accept_selectors,
                "🍪 Bouton refuser absent : bandeau cookies accepté.",
            ),
        ):
            for scope in scopes:
                for selector in selectors:
                    try:
                        btn = scope.locator(selector).first
                        if await btn.is_visible(timeout=350):
                            await btn.click()
                            await page.wait_for_timeout(400)
                            print(log_msg)
                            return action_name
                    except Exception:
                        continue

        return "none"

    async def _wait_until_window_closed(
        self,
        context: BrowserContext,
        page: Page,
        max_wait_seconds: int,
    ) -> None:
        """Keep browser open until user closes window or safety timeout expires.

        Args:
            context: Active Playwright browser context.
            page: Main application page.
            max_wait_seconds: Tunable maximum wait ceiling in seconds.
        """
        elapsed = 0
        while elapsed < max_wait_seconds:
            try:
                if page.is_closed() or not context.pages:
                    break
            except Exception:
                break
            await asyncio.sleep(1.0)
            elapsed += 1

    async def open_and_prefill_job(
        self,
        url: str,
        candidate: CandidateDetails | None = None,
        cover_letter: str | None = None,
        keep_open_seconds: int = 1800,
    ) -> bool:
        """Open ATS job application in visible browser, prefill and wait for user.

        Args:
            url: Job offer application URL.
            candidate: Candidate profile details (defaults to standard).
            cover_letter: Tailored cover letter to populate if requested.
            keep_open_seconds: Max wait ceiling in seconds (default 1800s = 30m).

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
                await page.wait_for_timeout(800)
                await self._handle_cookie_banner(page)

                # Tentative d'auto-remplissage des champs standards
                filled = await self._fill_common_fields(
                    page=page,
                    candidate=details,
                    cover_letter=cover_letter,
                )
                print(
                    f"✓ [Playwright] {filled} champ(s) pré-rempli(s). "
                    f"Fenêtre ouverte jusqu'à fermeture manuelle "
                    f"(max {keep_open_seconds // 60} min)."
                )

                # Maintien tant que l'utilisateur n'a pas fermé la fenêtre
                if not self.headless and keep_open_seconds > 0:
                    await self._wait_until_window_closed(
                        context=context,
                        page=page,
                        max_wait_seconds=keep_open_seconds,
                    )
                return True

            except Exception as err:
                print(f"⚠️ [Playwright] Erreur lors de la navigation : {err}")
                return False
            finally:
                try:
                    await context.close()
                except Exception:
                    pass

    async def _extract_field_label(self, scope: Locator, element: Locator) -> str:
        """Resolve the human-readable question label for a form control.

        Args:
            scope: Modal or form container locator.
            element: Target input, select, or textarea locator.

        Returns:
            str: Extracted label text, or empty string if unavailable.
        """
        try:
            aria = await element.get_attribute("aria-label")
            if aria and aria.strip():
                return aria.strip()

            elem_id = await element.get_attribute("id")
            if elem_id:
                label_el = scope.locator(f"label[for='{elem_id}']").first
                if await label_el.count() > 0:
                    label_txt = (await label_el.inner_text()).strip()
                    if label_txt:
                        return label_txt

            # Recherche d'un label parent ou conteneur immédiat
            parent_label = element.locator(
                "xpath=ancestor::div[contains(@class,'form-element') "
                "or contains(@class,'form-group') or contains(@class,'field')][1]"
                "//label"
            ).first
            if await parent_label.count() > 0:
                txt = (await parent_label.inner_text()).strip()
                if txt:
                    return txt

            placeholder = await element.get_attribute("placeholder")
            if placeholder and placeholder.strip():
                return placeholder.strip()
        except Exception:
            pass
        return ""

    async def _solve_modal_questions(
        self,
        page: Page,
        job_context: str = "",
    ) -> tuple[int, list[str]]:
        """Detect and answer screening questions inside an open modal step.

        Args:
            page: Active Playwright page.
            job_context: Optional title and company context for Gemini.

        Returns:
            tuple[int, list[str]]: (filled_count, list of unanswered labels).
        """
        filled = 0
        unanswered: list[str] = []

        modal_loc = page.locator(
            ".jobs-easy-apply-modal, [role='dialog'], form"
        ).first
        scope = modal_loc if await modal_loc.count() > 0 else page.locator("body")

        # 1. Champs texte et numériques non encore remplis
        text_inputs = await scope.locator(
            "input[type='text']:visible, input[type='number']:visible"
        ).all()
        for inp in text_inputs:
            try:
                if (await inp.input_value()).strip():
                    continue
                label = await self._extract_field_label(scope, inp)
                if not label:
                    continue

                input_type_attr = (
                    await inp.get_attribute("type") or "text"
                ).lower()
                is_num = input_type_attr == "number" or any(
                    tok in label.lower()
                    for tok in ("combien", "années", "years", "nombre", "how many")
                )
                field_type = "number" if is_num else "text"

                answer = await self.question_solver.solve_question(
                    question=label,
                    field_type=field_type,
                    job_context=job_context,
                )
                if answer is not None:
                    await inp.fill(answer)
                    filled += 1
                else:
                    unanswered.append(label)
            except Exception:
                continue

        # 2. Listes déroulantes (<select>)
        selects = await scope.locator("select:visible").all()
        for sel in selects:
            try:
                current_val = (await sel.input_value()).strip().lower()
                if current_val and not any(
                    ph in current_val
                    for ph in ("select", "sélection", "choisir", "--")
                ):
                    continue

                label = await self._extract_field_label(scope, sel)
                option_els = await sel.locator("option").all()
                options: list[str] = []
                for opt in option_els:
                    opt_txt = (await opt.inner_text()).strip()
                    opt_val = (await opt.get_attribute("value") or "").strip()
                    if not opt_txt or not opt_val:
                        continue
                    if any(
                        ph in opt_txt.lower()
                        for ph in ("sélectionner", "select an option", "choisir")
                    ):
                        continue
                    options.append(opt_txt)

                if not label or not options:
                    continue

                answer = await self.question_solver.solve_question(
                    question=label,
                    field_type="select",
                    options=options,
                    job_context=job_context,
                )
                if answer is not None:
                    await sel.select_option(label=answer)
                    filled += 1
                else:
                    unanswered.append(label)
            except Exception:
                continue

        # 3. Groupes de boutons radio (<fieldset>)
        fieldsets = await scope.locator("fieldset:visible").all()
        for fs in fieldsets:
            try:
                if await fs.locator("input[type='radio']:checked").count() > 0:
                    continue
                legend_el = fs.locator("legend, span, p").first
                question = (
                    (await legend_el.inner_text()).strip()
                    if await legend_el.count() > 0
                    else ""
                )
                label_els = await fs.locator("label:visible").all()
                radio_options = [
                    (await lbl.inner_text()).strip()
                    for lbl in label_els
                    if (await lbl.inner_text()).strip()
                ]
                if not question or not radio_options:
                    continue

                answer = await self.question_solver.solve_question(
                    question=question,
                    field_type="radio",
                    options=radio_options,
                    job_context=job_context,
                )
                if answer is not None:
                    target_lbl = fs.locator(
                        f"label:has-text('{answer}')"
                    ).first
                    if await target_lbl.count() > 0:
                        await target_lbl.click()
                        filled += 1
                    else:
                        unanswered.append(question)
                else:
                    unanswered.append(question)
            except Exception:
                continue

        return filled, unanswered

    async def _is_terminal_submit_visible(self, page: Page) -> bool:
        """Check whether the final application submit button is visible.

        Args:
            page: Active Playwright page.

        Returns:
            bool: True if a terminal Submit button is currently visible.
        """
        submit_btn = page.locator(
            "button:has-text('Envoyer la candidature'), "
            "button:has-text('Submit application'), "
            "button[aria-label*='Submit application' i], "
            "button[aria-label*='Envoyer la candidature' i]"
        ).first
        try:
            return await submit_btn.is_visible(timeout=500)
        except Exception:
            return False

    async def _click_next_step_button(self, page: Page) -> str:
        """Click the progression button ('Review' or 'Next') if available.

        Args:
            page: Active Playwright page.

        Returns:
            str: 'review', 'next', or 'none' if no progression button found.
        """
        candidates: list[tuple[str, str]] = [
            (
                "review",
                "button:has-text('Vérifier'), button:has-text('Review'), "
                "button[aria-label*='Review' i], "
                "button[aria-label*='Vérifier' i]",
            ),
            (
                "next",
                "button:has-text('Suivant'), button:has-text('Next'), "
                "button:has-text('Continuer'), button:has-text('Continue'), "
                "button[aria-label*='Continue' i], "
                "button[aria-label*='Suivant' i]",
            ),
        ]
        for action_name, selector in candidates:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500) and await btn.is_enabled():
                    await btn.click()
                    return action_name
            except Exception:
                continue
        return "none"

    async def _has_validation_errors(self, page: Page) -> bool:
        """Detect visible inline form validation error banners.

        Args:
            page: Active Playwright page.

        Returns:
            bool: True if blocking validation errors are visible.
        """
        err_locator = page.locator(
            ".artdeco-inline-feedback--error:visible, "
            "[role='alert']:visible, "
            ".error-message:visible, "
            ".fb-dash-form-element__error-field:visible"
        ).first
        try:
            return await err_locator.is_visible(timeout=400)
        except Exception:
            return False

    async def _process_modal_steps(
        self,
        page: Page,
        details: CandidateDetails,
        cover_letter: str | None = None,
        job_context: str = "",
        max_steps: int = 8,
        step_wait_ms: int = 1500,
    ) -> ApplicationModalResult:
        """Iterate through multi-step modal screens stopping before final submit.

        Args:
            page: Active Playwright page displaying the modal.
            details: Candidate contact details.
            cover_letter: Optional tailored cover letter text.
            job_context: Job title and company name for Gemini context.
            max_steps: Safety limit of step transitions to prevent loops.
            step_wait_ms: Delay in milliseconds after each step transition.

        Returns:
            ApplicationModalResult: Structured summary of progress and status.
        """
        total_filled = 0
        all_unanswered: list[str] = []
        steps_completed = 0
        final_status = "ready_for_manual_submit"

        for step in range(1, max_steps + 1):
            steps_completed = step
            filled_common = await self._fill_common_fields(
                page=page,
                candidate=details,
                cover_letter=cover_letter,
            )
            filled_q, step_unanswered = await self._solve_modal_questions(
                page=page,
                job_context=job_context,
            )
            total_filled += filled_common + filled_q
            for q in step_unanswered:
                if q not in all_unanswered:
                    all_unanswered.append(q)

            # Option A (HITL strict) : arrêt sécurisé avant envoi final
            if await self._is_terminal_submit_visible(page):
                print(
                    f"⏸️ [Playwright] Étape finale atteinte ({step} étape(s), "
                    f"{total_filled} champ(s) rempli(s)). En attente de "
                    "validation humaine."
                )
                final_status = "ready_for_manual_submit"
                break

            action = await self._click_next_step_button(page)
            if action == "none":
                final_status = (
                    "blocked_on_validation"
                    if all_unanswered
                    else "ready_for_manual_submit"
                )
                break

            await page.wait_for_timeout(step_wait_ms)

            # Si le bouton Suivant a déclenché une erreur bloquante
            if await self._has_validation_errors(page):
                print(
                    "⚠️ [Playwright] Blocage détecté sur erreur de validation "
                    f"à l'étape {step}."
                )
                final_status = "blocked_on_validation"
                break

        return ApplicationModalResult(
            status=final_status,
            steps_completed=steps_completed,
            fields_filled=total_filled,
            unanswered_questions=all_unanswered,
        )

    async def apply_multi_step_modal(
        self,
        url: str,
        candidate: CandidateDetails | None = None,
        cover_letter: str | None = None,
        job_context: str = "",
        screenshot_dir: Path | None = None,
        max_steps: int = 8,
        step_wait_ms: int = 1500,
        keep_open_seconds: int = 1800,
    ) -> ApplicationModalResult:
        """Open job URL, traverse multi-step Easy Apply modal, and pause for review.

        Never clicks the final submit button (strict Human-in-the-Loop).

        Args:
            url: Target job posting URL.
            candidate: Candidate profile contact information.
            cover_letter: Optional tailored cover letter to inject.
            job_context: Title and company context passed to question solver.
            screenshot_dir: Folder where confirmation screenshot is written.
            max_steps: Tunable cap on modal screens to traverse.
            step_wait_ms: Tunable pause between modal transitions.
            keep_open_seconds: Max seconds to leave visible browser open at end.

        Returns:
            ApplicationModalResult: Detailed result with screenshot path.
        """
        details = candidate or CandidateDetails()
        output_dir = screenshot_dir or (Path("data") / "screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        screenshot_path = output_dir / f"easy_apply_{timestamp}.png"

        print(f"💼 [Playwright] Démarrage du flux multi-étapes : {url}")

        async with async_playwright() as p:
            context: BrowserContext = await p.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                slow_mo=self.slow_mo,
            )

            try:
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(800)
                await self._handle_cookie_banner(page)

                # Détection du bouton de candidature initiale
                apply_button = page.locator(
                    "button.jobs-apply-button, "
                    "button:has-text('Candidature simplifiée'), "
                    "button:has-text('Easy Apply'), "
                    "button:has-text('Postuler'), "
                    "a:has-text('Postuler'), "
                    "button:has-text('Apply now')"
                ).first

                if not await apply_button.is_visible(timeout=5000):
                    print("⚠️ Bouton de candidature non détecté sur la page.")
                    await page.screenshot(path=str(screenshot_path))
                    return ApplicationModalResult(
                        status="apply_button_not_found",
                        screenshot_path=screenshot_path,
                    )

                await apply_button.click()
                await page.wait_for_timeout(step_wait_ms)

                result = await self._process_modal_steps(
                    page=page,
                    details=details,
                    cover_letter=cover_letter,
                    job_context=job_context,
                    max_steps=max_steps,
                    step_wait_ms=step_wait_ms,
                )

                await page.screenshot(path=str(screenshot_path))
                result.screenshot_path = screenshot_path
                print(f"📸 Capture d'écran enregistrée : {screenshot_path}")

                # Maintien de la fenêtre jusqu'à fermeture manuelle (Option A)
                if not self.headless and keep_open_seconds > 0:
                    print(
                        "⏳ Fenêtre maintenue ouverte jusqu'à fermeture "
                        f"manuelle (max {keep_open_seconds // 60} min)..."
                    )
                    await self._wait_until_window_closed(
                        context=context,
                        page=page,
                        max_wait_seconds=keep_open_seconds,
                    )

                return result

            except Exception as err:
                print(f"⚠️ [Playwright] Erreur flux multi-étapes : {err}")
                return ApplicationModalResult(
                    status="error",
                    screenshot_path=(
                        screenshot_path if screenshot_path.exists() else None
                    ),
                )
            finally:
                try:
                    await context.close()
                except Exception:
                    pass

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
        result = await self.apply_multi_step_modal(
            url=url,
            candidate=candidate,
            screenshot_dir=screenshot_dir,
        )
        return result.screenshot_path
