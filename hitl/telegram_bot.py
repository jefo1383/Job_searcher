"""Human-in-the-Loop Telegram notification and interactive action module."""

import asyncio
from collections.abc import Awaitable, Callable
import html
import re
import unicodedata
from typing import Any
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from automation.browser_runner import BrowserRunner, CandidateDetails
from config.settings import get_settings
from core.analyzer import JobAnalysisResult
from core.profile_builder import ProfileBuilder
from db.database import JobDatabase
from db.models import JobRecord, JobStatus
from scrapers.la_bonne_alternance import (
    LaBonneAlternanceScraper,
    SpontaneousRecruiter,
)

# Ordre d'affichage et libellés des catégories de contrat pour /pending
CONTRACT_CATEGORY_ORDER: tuple[str, ...] = (
    "alternance",
    "stage",
    "cdi",
    "cdd",
    "interim",
    "freelance",
    "autre",
)

CONTRACT_LABELS: dict[str, str] = {
    "alternance": "🎓 Alternance",
    "stage": "🧑‍🎓 Stage",
    "cdi": "💼 CDI",
    "cdd": "📄 CDD",
    "interim": "⏱️ Intérim",
    "freelance": "🚀 Freelance",
    "autre": "📌 Autre / Non précisé",
}

# Clés structurées éventuelles dans raw_payload issues des scrapers
PAYLOAD_CONTRACT_KEYS: tuple[str, ...] = (
    "contract_type",
    "contractType",
    "employmentType",
    "employment_type",
    "contract",
    "contrat",
    "job_type",
    "type",
)

# Expressions ordonnées (alternance/stage avant CDI pour éviter les faux positifs)
CONTRACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "alternance",
        re.compile(
            r"\b(alternance|alternant|apprentissage|apprenti|"
            r"contrat\s+pro(?:fessionnalisation)?|apprenticeship|work[\s-]study)\b"
        ),
    ),
    (
        "stage",
        re.compile(r"\b(stage|stagiaire|internship|intern)\b"),
    ),
    (
        "cdd",
        re.compile(r"\b(cdd|contrat\s+a\s+duree\s+determinee|fixed[\s-]term)\b"),
    ),
    (
        "interim",
        re.compile(
            r"\b(interim|interimaire|mission\s+d\s*interim|temporary)\b"
        ),
    ),
    (
        "freelance",
        re.compile(
            r"\b(freelance|independant|portage|prestation|"
            r"consultant\s+externe)\b"
        ),
    ),
    (
        "cdi",
        re.compile(
            r"\b(cdi|contrat\s+a\s+duree\s+indeterminee|"
            r"permanent|regular\s+employee)\b"
        ),
    ),
)


# Signature du callback d'exécution d'une session Indeed interactive
IndeedRunnerCallback = Callable[
    [Callable[[str], Awaitable[bool]]],
    Awaitable[tuple[int, int]],
]


class JobHunterBot:
    """Telegram bot interface for interactive job review and actions.

    Attributes:
        db: Database persistence manager instance.
        chat_id: Authorized Telegram chat identifier.
        application: Telegram Application instance.
    """

    def __init__(
        self,
        db: JobDatabase,
        bot_token: str | None = None,
        chat_id: int | None = None,
        browser_runner: BrowserRunner | None = None,
        profile_builder: ProfileBuilder | None = None,
        indeed_callback: IndeedRunnerCallback | None = None,
        lba_scraper: LaBonneAlternanceScraper | None = None,
    ) -> None:
        """Initialize the bot with settings and command handlers.

        Args:
            db: Initialized JobDatabase instance.
            bot_token: Telegram Bot API token (defaults to settings).
            chat_id: Telegram target user chat ID (defaults to settings).
            browser_runner: Browser automation runner for HITL actions.
            profile_builder: Profile generator for resume sync.
            indeed_callback: Optional interactive Indeed orchestrator hook.
            lba_scraper: Optional La bonne alternance scraper for /spontanees.

        Raises:
            ValueError: If bot_token is empty or invalid.
        """
        settings = get_settings()

        resolved_token = (
            bot_token if bot_token is not None
            else settings.telegram_bot_token.get_secret_value()
        )
        if not resolved_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set. Please provide it in .env "
                "or pass it to JobHunterBot."
            )

        self.db = db
        self.chat_id = (
            chat_id if chat_id is not None else settings.telegram_chat_id
        )
        self.browser_runner = browser_runner or BrowserRunner()
        self.profile_builder = profile_builder
        self.indeed_callback = indeed_callback
        self.lba_scraper = lba_scraper or LaBonneAlternanceScraper()
        self.candidate_details = CandidateDetails()

        # Coordination d'une session Indeed interactive lancée via /indeed
        self._indeed_lock = asyncio.Lock()
        self._indeed_ready_event: asyncio.Event | None = None
        self._indeed_cancelled: bool = False

        # Construction de l'application Telegram avec initialisation de la base
        self.application: Application[Any, Any, Any, Any, Any, Any] = (
            ApplicationBuilder()
            .token(resolved_token)
            .post_init(self._post_init)
            .build()
        )

        self._setup_handlers()

    async def _post_init(
        self, application: Application[Any, Any, Any, Any, Any, Any]
    ) -> None:
        """Execute post-initialization tasks, ensuring tables exist.

        Args:
            application: Telegram application instance.
        """
        # Initialisation automatique du schéma SQLite au démarrage du bot
        await self.db.init_db()

    def _setup_handlers(self) -> None:
        """Register Telegram command and callback query handlers."""
        # Commandes de supervision
        self.application.add_handler(
            CommandHandler("start", self._cmd_start)
        )
        self.application.add_handler(
            CommandHandler("stats", self._cmd_stats)
        )
        self.application.add_handler(
            CommandHandler("pending", self._cmd_pending)
        )
        self.application.add_handler(
            CommandHandler("accepted", self._cmd_accepted)
        )
        self.application.add_handler(
            CommandHandler("indeed", self._cmd_indeed)
        )
        self.application.add_handler(
            CommandHandler("spontanees", self._cmd_spontanees)
        )
        self.application.add_handler(
            CommandHandler("update_profile", self._cmd_update_profile)
        )

        # Gestionnaire de clics sur boutons interactifs
        self.application.add_handler(
            CallbackQueryHandler(self._handle_action_callback)
        )

        # Gestionnaire d'erreurs global
        self.application.add_error_handler(self._handle_error)

    async def _handle_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Log unhandled errors occurring in bot dispatch.

        Args:
            update: Object representing the update that caused error.
            context: Context containing the exception.
        """
        print(f"⚠️ Telegram bot error: {context.error}")

    def _is_authorized(self, user_chat_id: int) -> bool:
        """Verify whether an incoming request originates from authorized user.

        Args:
            user_chat_id: Chat identifier of the sender.

        Returns:
            bool: True if authorized, False otherwise.
        """
        if self.chat_id is None:
            # Si aucun chat_id n'est fixé, on accepte pour l'amorce
            return True
        return user_chat_id == self.chat_id

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        if not update.effective_chat or not update.message:
            return

        chat_id = update.effective_chat.id
        user_name = (
            update.effective_user.first_name
            if update.effective_user
            else "Cher candidat"
        )

        welcome_text = (
            f"👋 Bonjour <b>{html.escape(user_name)}</b> !\n\n"
            f"🤖 Je suis votre <b>Job Hunter Bot</b>.\n"
            f"Votre chat ID est : <code>{chat_id}</code>\n\n"
            "📋 <b>Commandes disponibles :</b>\n"
            "• /stats : Statistiques des offres en base de données.\n"
            "• /pending : Offres en attente triées par type de contrat.\n"
            "• /accepted : Offres acceptées en cours de candidature.\n"
            "• /indeed : Ouvre Indeed dans Chrome pour scroller manuellement.\n"
            "• /spontanees : Entreprises IT à contacter en alternance (<25 km).\n"
            "• /update_profile : Analyse le CV et régénère profile.md.\n\n"
            "Dès qu'une offre dépasse votre seuil de pertinence, vous "
            "recevrez une alerte avec 3 boutons d'action 1-clic."
        )

        await update.message.reply_text(
            welcome_text, parse_mode=ParseMode.HTML
        )

    async def _cmd_update_profile(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /update_profile command to regenerate candidate profile."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        if self.profile_builder is None:
            await update.message.reply_text(
                "⚠️ Module ProfileBuilder non disponible."
            )
            return

        status_msg = await update.message.reply_text(
            "⏳ Analyse de votre CV dans Profile/ et régénération de "
            "profile.md via Gemini Flash..."
        )

        try:
            regenerated = await self.profile_builder.sync_profile(force=True)
            if regenerated:
                await status_msg.edit_text(
                    "✅ <b>Profil candidat régénéré avec succès !</b>\n"
                    "Le fichier <code>config/profile.md</code> a été mis à "
                    "jour à partir de votre CV et de vos critères.",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await status_msg.edit_text(
                    "ℹ️ Aucun fichier de CV (.pdf ou .docx) trouvé dans le "
                    "dossier <code>Profile/</code>.",
                    parse_mode=ParseMode.HTML,
                )
        except Exception as err:
            await status_msg.edit_text(
                f"❌ Erreur lors de la régénération du profil : {err}"
            )

    async def _cmd_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /stats command."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        counts = await self.db.count_jobs_by_status()
        total = sum(counts.values())
        evaluating_count = counts.get(JobStatus.EVALUATING, 0)

        stats_text = (
            "📊 <b>Statistiques du Job Hunter</b>\n\n"
            f"• <b>Total collecté :</b> {total}\n"
            f"• 🆕 Nouvelles : <b>{counts.get(JobStatus.NEW, 0)}</b>\n"
            + (
                f"• 🧠 En cours d'évaluation : <b>{evaluating_count}</b>\n"
                if evaluating_count > 0
                else ""
            )
            + f"• ⏳ En attente de revue : "
            f"<b>{counts.get(JobStatus.PENDING_REVIEW, 0)}</b>\n"
            f"• ✅ Acceptées : <b>{counts.get(JobStatus.ACCEPTED, 0)}</b>\n"
            f"• 🚀 Postulées : <b>{counts.get(JobStatus.APPLIED, 0)}</b>\n"
            f"• ❌ Rejetées / Ignorées : "
            f"<b>{counts.get(JobStatus.REJECTED, 0)}</b>"
        )

        await update.message.reply_text(stats_text, parse_mode=ParseMode.HTML)

    def _resolve_apply_context(self, job: JobRecord) -> tuple[bool, str]:
        """Determine Easy Apply capability and target URL from a job record.

        Args:
            job: Target job offer record.

        Returns:
            tuple[bool, str]: Pair of (is_easy_apply, target_url).
        """
        # Détection si candidature simplifiée ou redirection ATS externe
        is_easy = True
        apply_url: str | None = None
        if job.raw_payload:
            if "is_easy_apply" in job.raw_payload:
                is_easy = bool(job.raw_payload.get("is_easy_apply"))
            raw_apply = job.raw_payload.get("apply_url")
            if isinstance(raw_apply, str) and raw_apply:
                apply_url = raw_apply

        return is_easy, apply_url or job.url

    @staticmethod
    def _normalize_contract_text(text: str) -> str:
        """Normalize text by lowercasing and stripping accents for matching.

        Args:
            text: Raw contract or job text.

        Returns:
            str: Normalized lowercase ASCII-friendly text.
        """
        decomposed = unicodedata.normalize("NFKD", text)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        return stripped.lower()

    @classmethod
    def _match_contract_text(cls, raw_text: str) -> str | None:
        """Match normalized text against ordered contract type patterns.

        Args:
            raw_text: Source text snippet to inspect.

        Returns:
            str | None: Canonical contract category key, or None if unmatched.
        """
        normalized = cls._normalize_contract_text(raw_text)
        for category, pattern in CONTRACT_PATTERNS:
            if pattern.search(normalized):
                return category
        return None

    def _detect_contract_category(self, job: JobRecord) -> str:
        """Detect contract category using DB field, payload, then text fallback.

        Args:
            job: Persisted job offer record.

        Returns:
            str: Canonical contract key (e.g. 'alternance', 'cdi', 'autre').
        """
        # 1. Lecture prioritaire de la colonne contract_type (extraite par Gemini)
        if job.contract_type and job.contract_type.strip():
            matched = self._match_contract_text(job.contract_type)
            if matched is not None:
                return matched

        # 2. Inspection des métadonnées structurées (raw_payload et raw_item)
        if job.raw_payload:
            payload_sources: list[dict[str, Any]] = [job.raw_payload]
            nested_item = job.raw_payload.get("raw_item")
            if isinstance(nested_item, dict):
                payload_sources.append(nested_item)

            for source in payload_sources:
                for key in PAYLOAD_CONTRACT_KEYS:
                    raw_val = source.get(key)
                    if isinstance(raw_val, list):
                        text_val = " ".join(str(v) for v in raw_val if v)
                    elif isinstance(raw_val, str):
                        text_val = raw_val
                    else:
                        continue
                    if text_val.strip():
                        matched = self._match_contract_text(text_val)
                        if matched is not None:
                            return matched

            # Cas Workday (champs bulletFields contenant ex: 'Regular Employee')
            bullet_fields = job.raw_payload.get("bulletFields")
            if isinstance(bullet_fields, list):
                matched = self._match_contract_text(
                    " ".join(str(b) for b in bullet_fields if b)
                )
                if matched is not None:
                    return matched

        # 3. Fallback déterministe sur le titre puis la description
        title_match = self._match_contract_text(job.title)
        if title_match is not None:
            return title_match

        desc_match = self._match_contract_text(job.description[:2000])
        if desc_match is not None:
            return desc_match

        return "autre"

    def _build_pending_menu(
        self, pending_jobs: list[JobRecord]
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        """Build category selection keyboard for pending review offers.

        Args:
            pending_jobs: List of PENDING_REVIEW job records.

        Returns:
            tuple[str, InlineKeyboardMarkup | None]: Menu HTML text and markup.
        """
        if not pending_jobs:
            return (
                "✨ Aucune offre en attente de revue pour le moment !",
                None,
            )

        # Comptage par catégorie détectée
        counts: dict[str, int] = {}
        for job in pending_jobs:
            category = self._detect_contract_category(job)
            counts[category] = counts.get(category, 0) + 1

        # Affichage uniquement des catégories ayant au moins 1 offre en attente
        category_buttons: list[InlineKeyboardButton] = []
        for cat_key in CONTRACT_CATEGORY_ORDER:
            count = counts.get(cat_key, 0)
            if count > 0:
                label = CONTRACT_LABELS.get(cat_key, cat_key.upper())
                category_buttons.append(
                    InlineKeyboardButton(
                        f"{label} ({count})",
                        callback_data=f"pending_cat:{cat_key}",
                    )
                )

        # Regroupement sur 2 colonnes + bouton global "Toutes les offres"
        keyboard: list[list[InlineKeyboardButton]] = [
            category_buttons[i : i + 2]
            for i in range(0, len(category_buttons), 2)
        ]
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"📋 Toutes les offres ({len(pending_jobs)})",
                    callback_data="pending_cat:all",
                )
            ]
        )

        menu_text = (
            f"⏳ <b>{len(pending_jobs)} offre(s) en attente de revue</b>\n"
            "Choisissez une catégorie de contrat à afficher :"
        )
        return menu_text, InlineKeyboardMarkup(keyboard)

    def _build_accepted_markup(
        self,
        job_id: int,
        offer_url: str | None = None,
    ) -> InlineKeyboardMarkup:
        """Create action keyboard for ACCEPTED (in-progress) applications.

        Args:
            job_id: SQLite primary key of the job.
            offer_url: Target ATS or job posting URL.

        Returns:
            InlineKeyboardMarkup: Follow-up buttons to finish, reopen, or drop.
        """
        keyboard: list[list[InlineKeyboardButton]] = [
            [
                InlineKeyboardButton(
                    "✅ Candidature envoyée",
                    callback_data=f"mark_applied:{job_id}",
                ),
                InlineKeyboardButton(
                    "📝 Ré-ouvrir l'ATS",
                    callback_data=f"fill:{job_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "↩️ Remettre en attente",
                    callback_data=f"reopen:{job_id}",
                ),
                InlineKeyboardButton(
                    "❌ Ignorer",
                    callback_data=f"reject:{job_id}",
                ),
            ],
        ]
        if offer_url:
            keyboard.append(
                [InlineKeyboardButton("🌐 Voir l'offre", url=offer_url)]
            )
        return InlineKeyboardMarkup(keyboard)

    def format_pending_message(
        self,
        job: JobRecord,
        header_title: str = "⏳ OFFRE EN ATTENTE",
    ) -> str:
        """Format an interactive HTML card for a pending or accepted job.

        Args:
            job: Persisted job offer awaiting user decision or completion.
            header_title: Card header label.

        Returns:
            str: HTML formatted notification card.
        """
        score_txt = f"{job.score}/100" if job.score is not None else "N/A"
        contract_key = self._detect_contract_category(job)
        contract_label = CONTRACT_LABELS.get(contract_key, "Autre")
        safe_company = html.escape(job.company)
        safe_title = html.escape(job.title)
        safe_platform = html.escape(job.platform.upper())
        safe_url = html.escape(job.url)

        return "\n".join(
            [
                f"<b>{html.escape(header_title)} ({score_txt})</b>\n",
                f"🏢 <b>Entreprise :</b> {safe_company}",
                f"💼 <b>Poste :</b> {safe_title}",
                f"📑 <b>Contrat :</b> {html.escape(contract_label)}",
                f"🌐 <b>Source :</b> <a href='{safe_url}'>{safe_platform}</a>",
            ]
        )

    async def _cmd_accepted(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /accepted command to manage in-progress applications."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        # Nombre maximal d'offres en cours affichées (paramètre réglable)
        accepted_limit = 20
        accepted_jobs = await self.db.get_jobs_by_status(
            JobStatus.ACCEPTED, limit=accepted_limit
        )
        if not accepted_jobs:
            await update.message.reply_text(
                "✨ Aucune candidature en cours (statut ACCEPTED) !"
            )
            return

        await update.message.reply_text(
            f"🚧 <b>{len(accepted_jobs)} candidature(s) en cours :</b>",
            parse_mode=ParseMode.HTML,
        )

        for job in accepted_jobs:
            if job.id is None:
                continue
            _, target_url = self._resolve_apply_context(job)
            markup = self._build_accepted_markup(
                job_id=job.id,
                offer_url=target_url,
            )
            await update.message.reply_text(
                self.format_pending_message(
                    job, header_title="🚧 CANDIDATURE EN COURS"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
                disable_web_page_preview=True,
            )

    async def _cmd_pending(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /pending command by showing contract category buttons."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        # Nombre maximal d'offres agrégées dans le menu (paramètre réglable)
        pending_fetch_limit = 50
        pending_jobs = await self.db.get_jobs_by_status(
            JobStatus.PENDING_REVIEW, limit=pending_fetch_limit
        )
        menu_text, markup = self._build_pending_menu(pending_jobs)
        await update.message.reply_text(
            menu_text,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _handle_pending_callback(
        self, query: CallbackQuery, chat_id: int
    ) -> None:
        """Handle category selection and menu refresh callbacks for /pending.

        Args:
            query: Telegram callback query from inline keyboard.
            chat_id: Target authorized Telegram chat ID.
        """
        if not query.data:
            return

        # Paramètres réglables : profondeur d'agrégation et cartes par clic
        pending_fetch_limit = 50
        cards_display_limit = 10

        pending_jobs = await self.db.get_jobs_by_status(
            JobStatus.PENDING_REVIEW, limit=pending_fetch_limit
        )

        if query.data == "pending_menu":
            menu_text, markup = self._build_pending_menu(pending_jobs)
            try:
                await query.edit_message_text(
                    text=menu_text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
            except Exception:
                pass
            return

        cat_key = query.data.split(":", 1)[1]
        if cat_key == "all":
            filtered_jobs = pending_jobs
            cat_label = "📋 Toutes les offres"
        else:
            filtered_jobs = [
                job
                for job in pending_jobs
                if self._detect_contract_category(job) == cat_key
            ]
            cat_label = CONTRACT_LABELS.get(cat_key, cat_key.upper())

        back_markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🔙 Autres catégories",
                        callback_data="pending_menu",
                    )
                ]
            ]
        )

        if not filtered_jobs:
            try:
                await query.edit_message_text(
                    text=(
                        f"✨ Aucune offre en attente dans la catégorie "
                        f"<b>{html.escape(cat_label)}</b>."
                    ),
                    parse_mode=ParseMode.HTML,
                    reply_markup=back_markup,
                )
            except Exception:
                pass
            return

        shown_jobs = filtered_jobs[:cards_display_limit]
        try:
            await query.edit_message_text(
                text=(
                    f"📂 <b>{html.escape(cat_label)}</b> — "
                    f"{len(shown_jobs)}/{len(filtered_jobs)} "
                    "offre(s) affichée(s) :"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=back_markup,
            )
        except Exception:
            pass

        # Envoi d'une carte interactive par offre filtrée
        for job in shown_jobs:
            if job.id is None:
                continue
            is_easy, target_url = self._resolve_apply_context(job)
            markup = self._build_alert_markup(
                job_id=job.id,
                is_easy_apply=is_easy,
                offer_url=target_url,
            )
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=self.format_pending_message(job),
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
                disable_web_page_preview=True,
            )

    def _build_alert_markup(
        self,
        job_id: int,
        is_easy_apply: bool = True,
        offer_url: str | None = None,
    ) -> InlineKeyboardMarkup:
        """Create the inline interactive keyboard adapted to application mode.

        Args:
            job_id: SQLite primary key of the job.
            is_easy_apply: Whether the posting supports 1-click in-app apply.
            offer_url: Target URL of the job posting or external ATS portal.

        Returns:
            InlineKeyboardMarkup: Formed Telegram keyboard.
        """
        # Boutons d'action adaptés selon Easy Apply ou redirection ATS
        if is_easy_apply:
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🚀 Postuler Direct",
                        callback_data=f"apply:{job_id}",
                    ),
                    InlineKeyboardButton(
                        "📝 Pré-remplir",
                        callback_data=f"fill:{job_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "❌ Ignorer",
                        callback_data=f"reject:{job_id}",
                    ),
                ],
            ]
        else:
            action_row: list[InlineKeyboardButton] = [
                InlineKeyboardButton(
                    "📝 Pré-remplir sur l'ATS",
                    callback_data=f"fill:{job_id}",
                ),
            ]
            if offer_url:
                action_row.append(
                    InlineKeyboardButton(
                        "🌐 Voir l'offre",
                        url=offer_url,
                    )
                )
            keyboard = [
                action_row,
                [
                    InlineKeyboardButton(
                        "❌ Ignorer",
                        callback_data=f"reject:{job_id}",
                    ),
                ],
            ]
        return InlineKeyboardMarkup(keyboard)

    def format_alert_message(
        self,
        job: JobRecord,
        analysis: JobAnalysisResult,
        pitch: str | None = None,
    ) -> str:
        """Format an HTML alert notification card for Telegram.

        Args:
            job: Scraped job offer record.
            analysis: Structured evaluation from Gemini.
            pitch: Optional short motivation pitch.

        Returns:
            str: HTML formatted notification message.
        """
        safe_company = html.escape(job.company)
        safe_title = html.escape(job.title)
        safe_platform = html.escape(job.platform.upper())
        safe_reasoning = html.escape(analysis.reasoning)
        safe_url = html.escape(job.url)

        matches_list = (
            ", ".join(html.escape(m) for m in analysis.key_matches[:4])
            if analysis.key_matches
            else "Non spécifié"
        )

        contract_key = self._detect_contract_category(job)
        if contract_key == "autre" and analysis.contract_type:
            matched_ct = self._match_contract_text(analysis.contract_type)
            if matched_ct is not None:
                contract_key = matched_ct

        message_parts = [
            f"🎯 <b>NOUVELLE OFFRE QUALIFIÉE ({analysis.score}/100)</b>\n",
            f"🏢 <b>Entreprise :</b> {safe_company}",
            f"💼 <b>Poste :</b> {safe_title}",
        ]
        if contract_key != "autre":
            ct_label = CONTRACT_LABELS.get(contract_key, contract_key.upper())
            message_parts.append(
                f"📑 <b>Contrat :</b> {html.escape(ct_label)}"
            )
        message_parts.extend(
            [
                f"🌐 <b>Source :</b> <a href='{safe_url}'>{safe_platform}</a>",
                f"✅ <b>Points forts :</b> {matches_list}",
                f"💡 <b>Synthèse :</b> <i>{safe_reasoning}</i>",
            ]
        )

        if pitch:
            safe_pitch = html.escape(pitch)
            message_parts.append(f"\n📌 <b>Pitch suggéré :</b>\n<i>{safe_pitch}</i>")

        return "\n".join(message_parts)

    async def send_info_message(self, text: str) -> bool:
        """Send an informational HTML notice to the configured chat.

        Args:
            text: HTML formatted notification text.

        Returns:
            bool: True if delivered successfully, False otherwise.
        """
        if self.chat_id is None:
            return False
        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return True
        except Exception:
            return False

    async def _cmd_indeed(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /indeed command to launch visible browser for manual prep."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        if self.indeed_callback is None:
            await update.message.reply_text(
                "⚠️ Le mode /indeed interactif nécessite de lancer le bot via "
                "<code>main.py --bot-only</code> (ou <code>--indeed</code> en "
                "terminal).",
                parse_mode=ParseMode.HTML,
            )
            return

        if self._indeed_lock.locked():
            await update.message.reply_text(
                "⏳ Une session Indeed interactive est déjà en cours. "
                "Cliquez sur <b>Page prête</b> ou <b>Annuler</b> dans le message "
                "précédent.",
                parse_mode=ParseMode.HTML,
            )
            return

        await update.message.reply_text(
            "🖥️ Lancement de Chrome sur <code>fr.indeed.com</code>...\n"
            "Un bouton de confirmation va apparaître dès que la fenêtre est "
            "prête.",
            parse_mode=ParseMode.HTML,
        )
        asyncio.create_task(
            self._run_telegram_indeed_session(update.effective_chat.id)
        )

    async def _wait_for_telegram_indeed_ready(
        self,
        chat_id: int,
        target_url: str,
        timeout_seconds: float = 300.0,
    ) -> bool:
        """Prompt user via Telegram inline buttons and await ready/cancel."""
        event = asyncio.Event()
        self._indeed_ready_event = event
        self._indeed_cancelled = False

        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ Page prête, lancer le scroll",
                        callback_data="indeed:ready",
                    )
                ],
                [
                    InlineKeyboardButton(
                        "❌ Annuler",
                        callback_data="indeed:cancel",
                    )
                ],
            ]
        )

        safe_url = html.escape(target_url)
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=(
                "🌐 <b>Indeed ouvert dans Chrome !</b>\n\n"
                "1️⃣ Vérifiez/validez la page dans la fenêtre Chrome ouverte :\n"
                f"<code>{safe_url}</code>\n\n"
                "2️⃣ Une fois la liste d'offres visible (vous pouvez aussi "
                "ajuster votre recherche), cliquez ci-dessous :"
            ),
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
            disable_web_page_preview=True,
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            return not self._indeed_cancelled
        except TimeoutError:
            await self.send_info_message(
                "⌛ Délai d'attente expiré pour la session /indeed."
            )
            return False
        finally:
            self._indeed_ready_event = None

    async def _run_telegram_indeed_session(self, chat_id: int) -> None:
        """Execute interactive Indeed collection and notify summary."""
        if self.indeed_callback is None:
            return
        async with self._indeed_lock:
            try:
                async def waiter(url: str) -> bool:
                    return await self._wait_for_telegram_indeed_ready(
                        chat_id=chat_id,
                        target_url=url,
                    )

                collected, qualified = await self.indeed_callback(waiter)
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🏁 <b>Collecte Indeed terminée !</b>\n"
                        f"• Nouvelles offres uniques : <b>{collected}</b>\n"
                        f"• Offres qualifiées notifiées : <b>{qualified}</b>"
                    ),
                    parse_mode=ParseMode.HTML,
                )
            except Exception as err:
                safe_err = html.escape(str(err))
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Erreur durant la session /indeed : {safe_err}",
                    parse_mode=ParseMode.HTML,
                )

    @staticmethod
    def _build_spontanees_page(
        recruiters: list[SpontaneousRecruiter],
        offset: int = 0,
        page_size: int = 8,
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        """Format paginated Telegram message listing recruteurs_lba companies.

        Args:
            recruiters: Sorted list of spontaneous application targets.
            offset: Zero-based slice start index.
            page_size: Number of companies per page (tunable).

        Returns:
            tuple[str, InlineKeyboardMarkup | None]: HTML text and nav markup.
        """
        total = len(recruiters)
        if total == 0:
            return (
                "ℹ️ Aucune entreprise IT <code>recruteurs_lba</code> trouvée "
                "dans un rayon de 25 km (vérifiez <code>LBA_API_KEY</code>).",
                None,
            )

        safe_offset = max(0, min(offset, max(0, total - 1)))
        batch = recruiters[safe_offset:safe_offset + page_size]
        end_idx = safe_offset + len(batch)

        lines: list[str] = [
            f"🏢 <b>CANDIDATURES SPONTANÉES &lt;25 km ({safe_offset + 1}"
            f"-{end_idx}/{total})</b>",
            "<i>Entreprises IT identifiées par La bonne alternance :</i>\n",
        ]

        for idx, rec in enumerate(batch, start=safe_offset + 1):
            dist_txt = (
                f" • {rec.distance_km:.1f} km"
                if rec.distance_km is not None
                else ""
            )
            size_txt = f" • {html.escape(rec.size)}" if rec.size else ""
            sector = html.escape(rec.naf_label or "Secteur numérique")
            addr = html.escape(rec.address or "Bouches-du-Rhône")
            safe_apply = html.escape(rec.apply_url)

            links = [f"<a href='{safe_apply}'>Postuler via LBA</a>"]
            if rec.website and rec.website.startswith(("http://", "https://")):
                links.append(
                    f"<a href='{html.escape(rec.website)}'>Site web</a>"
                )

            lines.append(
                f"<b>{idx}. {html.escape(rec.company)}</b>"
                f"{dist_txt}{size_txt}\n"
                f"   🏷️ <i>{sector}</i>\n"
                f"   📍 {addr}\n"
                f"   🔗 {' | '.join(links)}"
            )

        nav_row: list[InlineKeyboardButton] = []
        if safe_offset > 0:
            prev_offset = max(0, safe_offset - page_size)
            nav_row.append(
                InlineKeyboardButton(
                    "⬅️ Précédentes",
                    callback_data=f"spontanees:{prev_offset}",
                )
            )
        if end_idx < total:
            nav_row.append(
                InlineKeyboardButton(
                    "➡️ Suivantes",
                    callback_data=f"spontanees:{end_idx}",
                )
            )

        rows: list[list[InlineKeyboardButton]] = []
        if nav_row:
            rows.append(nav_row)
        rows.append(
            [
                InlineKeyboardButton(
                    "🔄 Actualiser la liste",
                    callback_data="spontanees:refresh",
                )
            ]
        )
        return "\n\n".join(lines), InlineKeyboardMarkup(rows)

    async def _cmd_spontanees(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /spontanees command showing IT companies within 25 km."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        # Nombre d'entreprises affichées par page (paramètre réglable)
        page_size = 8
        recruiters = await self.lba_scraper.fetch_spontaneous_recruiters(
            it_only=True
        )
        msg_text, markup = self._build_spontanees_page(
            recruiters=recruiters,
            offset=0,
            page_size=page_size,
        )
        await update.message.reply_text(
            msg_text,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
            disable_web_page_preview=True,
        )

    async def _handle_spontanees_callback(self, query: CallbackQuery) -> None:
        """Handle pagination and refresh button clicks for /spontanees."""
        if not query.data:
            return
        param = query.data.split(":", 1)[1]
        force_refresh = param == "refresh"
        if force_refresh:
            offset = 0
        else:
            try:
                offset = int(param)
            except ValueError:
                return

        recruiters = await self.lba_scraper.fetch_spontaneous_recruiters(
            it_only=True,
            force_refresh=force_refresh,
        )
        msg_text, markup = self._build_spontanees_page(
            recruiters=recruiters,
            offset=offset,
            page_size=8,
        )
        try:
            await query.edit_message_text(
                text=msg_text,
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
                disable_web_page_preview=True,
            )
        except Exception:
            pass

    async def send_job_alert(
        self,
        job: JobRecord,
        analysis: JobAnalysisResult,
        pitch: str | None = None,
    ) -> bool:
        """Send an interactive job alert to the configured Telegram chat.

        Args:
            job: Populated JobRecord with an assigned id.
            analysis: Result from JobAnalyzer.
            pitch: Optional quick pitch text.

        Returns:
            bool: True if message sent successfully, False otherwise.
        """
        if self.chat_id is None:
            return False

        if job.id is None:
            return False

        is_easy, target_url = self._resolve_apply_context(job)
        text = self.format_alert_message(job, analysis, pitch)
        markup = self._build_alert_markup(
            job_id=job.id,
            is_easy_apply=is_easy,
            offer_url=target_url,
        )

        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
                disable_web_page_preview=True,
            )
            return True
        except Exception:
            return False

    async def _handle_action_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle inline button clicks and update job status accordingly."""
        query = update.callback_query
        if not query or not query.data:
            return

        # Accusé de réception immédiat pour retirer le sablier Telegram
        await query.answer()

        if not update.effective_chat or not self._is_authorized(
            update.effective_chat.id
        ):
            await query.edit_message_reply_markup(reply_markup=None)
            return

        # Routage dédié aux filtres de catégories de contrat /pending
        if query.data == "pending_menu" or query.data.startswith("pending_cat:"):
            await self._handle_pending_callback(query, update.effective_chat.id)
            return

        # Pagination et actualisation de /spontanees
        if query.data.startswith("spontanees:"):
            await self._handle_spontanees_callback(query)
            return

        # Confirmation ou annulation d'une session Indeed interactive
        if query.data in ("indeed:ready", "indeed:cancel"):
            is_cancel = query.data == "indeed:cancel"
            self._indeed_cancelled = is_cancel
            if self._indeed_ready_event is not None:
                self._indeed_ready_event.set()
            status_txt = (
                "❌ Session /indeed annulée."
                if is_cancel
                else "🔄 Scroll et extraction Indeed en cours..."
            )
            try:
                await query.edit_message_text(text=status_txt)
            except Exception:
                pass
            return

        # Parsing de la charge utile (action:job_id)
        try:
            action, job_id_str = query.data.split(":", 1)
            job_id = int(job_id_str)
        except ValueError:
            return

        job = await self.db.get_job_by_id(job_id)
        if not job:
            return

        # Application de la transition d'état dans SQLite
        status_label = ""
        next_markup: InlineKeyboardMarkup | None = None
        if action == "apply":
            await self.db.update_job_status(job_id, JobStatus.ACCEPTED)
            status_label = "🚀 <b>ACTION : Candidature Playwright en cours...</b>"
            asyncio.create_task(
                self._execute_apply_task(job, update.effective_chat.id)
            )
        elif action == "fill":
            await self.db.update_job_status(job_id, JobStatus.ACCEPTED)
            status_label = (
                "📝 <b>ACTION : Navigateur ouvert avec pré-remplissage !</b>"
            )
            _, target_url = self._resolve_apply_context(job)
            next_markup = self._build_accepted_markup(
                job_id=job_id,
                offer_url=target_url,
            )

            asyncio.create_task(
                self.browser_runner.open_and_prefill_job(
                    url=target_url,
                    candidate=self.candidate_details,
                )
            )
        elif action == "reopen":
            await self.db.update_job_status(job_id, JobStatus.PENDING_REVIEW)
            status_label = (
                "↩️ <b>ACTION : Offre remise en attente (/pending).</b>"
            )
            is_easy, target_url = self._resolve_apply_context(job)
            next_markup = self._build_alert_markup(
                job_id=job_id,
                is_easy_apply=is_easy,
                offer_url=target_url,
            )
        elif action == "reject":
            await self.db.update_job_status(job_id, JobStatus.REJECTED)
            status_label = "❌ <b>ACTION : Offre ignorée et rejetée.</b>"
        elif action == "mark_applied":
            await self.db.update_job_status(job_id, JobStatus.APPLIED)
            status_label = (
                "🎉 <b>ACTION : Candidature marquée comme envoyée !</b>"
            )

        # Mise à jour visuelle (texte ou légende photo) avec boutons de suivi
        is_photo_msg = isinstance(query.message, Message) and bool(
            query.message.photo
        )
        if is_photo_msg and isinstance(query.message, Message):
            current_caption = query.message.caption_html or ""
            updated_caption = (
                f"{current_caption}\n\n{status_label}"
                if current_caption and not current_caption.endswith(status_label)
                else (current_caption or status_label)
            )
            try:
                await query.edit_message_caption(
                    caption=updated_caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=next_markup,
                )
            except Exception:
                pass
            return

        current_text = (
            query.message.text_html
            if isinstance(query.message, Message) and query.message.text_html
            else ""
        )
        if current_text:
            updated_text = (
                current_text
                if current_text.endswith(status_label)
                else f"{current_text}\n\n{status_label}"
            )
        else:
            updated_text = status_label

        try:
            await query.edit_message_text(
                text=updated_text,
                parse_mode=ParseMode.HTML,
                reply_markup=next_markup,
                disable_web_page_preview=True,
            )
        except Exception:
            # En cas de timeout ou message inchangé
            pass

    async def _execute_apply_task(self, job: JobRecord, chat_id: int) -> None:
        """Run multi-step Easy Apply modal and send review screenshot.

        Stops before final submission (strict HITL) and prompts the user to
        confirm once submitted manually in Chrome.

        Args:
            job: Target job record.
            chat_id: Telegram chat ID for delivery.
        """
        result = await self.browser_runner.apply_multi_step_modal(
            url=job.url,
            candidate=self.candidate_details,
            job_context=f"{job.title} chez {job.company}",
        )

        # Construction de la légende selon l'état d'avancement dans la modale
        if result.status == "ready_for_manual_submit":
            header = (
                f"⏸️ <b>Prêt pour validation finale !</b>\n"
                f"📊 {result.steps_completed} étape(s) — "
                f"{result.fields_filled} champ(s) rempli(s).\n"
                "Vérifiez le navigateur Chrome et cliquez sur <b>Envoyer</b>."
            )
        elif result.status == "blocked_on_validation":
            missing_txt = ", ".join(
                html.escape(q) for q in result.unanswered_questions[:3]
            )
            header = (
                "⚠️ <b>Intervention manuelle requise</b>\n"
                f"Questions bloquantes : <i>{missing_txt or 'Champs requis'}</i>"
            )
        else:
            header = (
                "⚠️ <b>Bouton Easy Apply non détecté ou erreur</b>\n"
                "Vérifiez la page dans le navigateur."
            )

        caption_text = (
            f"{header}\n\n"
            f"🏢 {html.escape(job.company)} — {html.escape(job.title)}"
        )

        _, target_url = self._resolve_apply_context(job)
        confirm_markup = (
            self._build_accepted_markup(
                job_id=job.id,
                offer_url=target_url,
            )
            if job.id is not None
            else None
        )

        if result.screenshot_path and result.screenshot_path.exists():
            try:
                with open(result.screenshot_path, "rb") as photo:
                    await self.application.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=caption_text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=confirm_markup,
                    )
                return
            except Exception as err:
                print(f"⚠️ Échec d'envoi de la capture Telegram : {err}")

        # Fallback texte si aucune capture n'a pu être générée
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=caption_text,
                parse_mode=ParseMode.HTML,
                reply_markup=confirm_markup,
                disable_web_page_preview=True,
            )
        except Exception as err:
            print(f"⚠️ Échec d'envoi du message de statut Telegram : {err}")

    def run_polling(self) -> None:
        """Run the bot in blocking polling mode (convenience for CLI execution)."""
        print("🤖 Job Hunter Bot is running in polling mode. Press Ctrl+C to stop.")
        self.application.run_polling()


if __name__ == "__main__":
    settings = get_settings()
    database = JobDatabase(settings.database_path)
    bot = JobHunterBot(db=database)
    bot.run_polling()
