"""Human-in-the-Loop Telegram notification and interactive action module."""

import asyncio
import html
from typing import Any
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
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
    ) -> None:
        """Initialize the bot with settings and command handlers.

        Args:
            db: Initialized JobDatabase instance.
            bot_token: Telegram Bot API token (defaults to settings).
            chat_id: Telegram target user chat ID (defaults to settings).
            browser_runner: Browser automation runner for HITL actions.
            profile_builder: Profile generator for resume sync.

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
        self.candidate_details = CandidateDetails()

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
            "• /pending : Liste des offres en attente de décision.\n"
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

        stats_text = (
            "📊 <b>Statistiques du Job Hunter</b>\n\n"
            f"• <b>Total collecté :</b> {total}\n"
            f"• 🆕 Nouvelles : <b>{counts.get(JobStatus.NEW, 0)}</b>\n"
            f"• ⏳ En attente de revue : "
            f"<b>{counts.get(JobStatus.PENDING_REVIEW, 0)}</b>\n"
            f"• ✅ Acceptées : <b>{counts.get(JobStatus.ACCEPTED, 0)}</b>\n"
            f"• 🚀 Postulées : <b>{counts.get(JobStatus.APPLIED, 0)}</b>\n"
            f"• ❌ Rejetées / Ignorées : "
            f"<b>{counts.get(JobStatus.REJECTED, 0)}</b>"
        )

        await update.message.reply_text(stats_text, parse_mode=ParseMode.HTML)

    async def _cmd_pending(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /pending command to display unreviewed offers."""
        if not update.effective_chat or not update.message:
            return

        if not self._is_authorized(update.effective_chat.id):
            await update.message.reply_text("⛔ Accès non autorisé.")
            return

        pending_jobs = await self.db.get_jobs_by_status(
            JobStatus.PENDING_REVIEW, limit=10
        )
        if not pending_jobs:
            await update.message.reply_text(
                "✨ Aucune offre en attente de revue pour le moment !"
            )
            return

        msg_lines = [
            f"⏳ <b>{len(pending_jobs)} offre(s) en attente de revue :</b>\n"
        ]
        for job in pending_jobs:
            score_txt = f"{job.score}/100" if job.score is not None else "N/A"
            msg_lines.append(
                f"• <b>{html.escape(job.company)}</b> — "
                f"<a href='{html.escape(job.url)}'>{html.escape(job.title)}</a> "
                f"(Score: <b>{score_txt}</b>)"
            )

        await update.message.reply_text(
            "\n".join(msg_lines),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    def _build_alert_markup(self, job_id: int) -> InlineKeyboardMarkup:
        """Create the 3-button inline interactive keyboard.

        Args:
            job_id: SQLite primary key of the job.

        Returns:
            InlineKeyboardMarkup: Formed Telegram keyboard.
        """
        # Boutons d'action conformes aux spécifications
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

        message_parts = [
            f"🎯 <b>NOUVELLE OFFRE QUALIFIÉE ({analysis.score}/100)</b>\n",
            f"🏢 <b>Entreprise :</b> {safe_company}",
            f"💼 <b>Poste :</b> {safe_title}",
            f"🌐 <b>Source :</b> <a href='{safe_url}'>{safe_platform}</a>",
            f"✅ <b>Points forts :</b> {matches_list}",
            f"💡 <b>Synthèse :</b> <i>{safe_reasoning}</i>",
        ]

        if pitch:
            safe_pitch = html.escape(pitch)
            message_parts.append(f"\n📌 <b>Pitch suggéré :</b>\n<i>{safe_pitch}</i>")

        return "\n".join(message_parts)

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

        text = self.format_alert_message(job, analysis, pitch)
        markup = self._build_alert_markup(job.id)

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
            asyncio.create_task(
                self.browser_runner.open_and_prefill_job(
                    url=job.url,
                    candidate=self.candidate_details,
                )
            )
        elif action == "reject":
            await self.db.update_job_status(job_id, JobStatus.REJECTED)
            status_label = "❌ <b>ACTION : Offre ignorée et rejetée.</b>"

        # Mise à jour visuelle du message pour retirer les boutons
        current_text = (
            query.message.text_html
            if isinstance(query.message, Message) and query.message.text_html
            else ""
        )
        if current_text:
            updated_text = f"{current_text}\n\n{status_label}"
        else:
            updated_text = status_label

        try:
            await query.edit_message_text(
                text=updated_text,
                parse_mode=ParseMode.HTML,
                reply_markup=None,
                disable_web_page_preview=True,
            )
        except Exception:
            # En cas de timeout ou message inchangé
            pass

    async def _execute_apply_task(self, job: JobRecord, chat_id: int) -> None:
        """Run Easy Apply flow and send back confirmation screenshot.

        Args:
            job: Target job record.
            chat_id: Telegram chat ID for delivery.
        """
        screenshot_path = await self.browser_runner.apply_linkedin_easy_apply(
            url=job.url,
            candidate=self.candidate_details,
        )
        if screenshot_path and screenshot_path.exists():
            try:
                with open(screenshot_path, "rb") as photo:
                    await self.application.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=(
                            f"📸 <b>Confirmation Easy Apply</b>\n"
                            f"🏢 {html.escape(job.company)} — "
                            f"{html.escape(job.title)}"
                        ),
                        parse_mode=ParseMode.HTML,
                    )
            except Exception as err:
                print(f"⚠️ Échec d'envoi de la capture Telegram : {err}")

        if job.id is not None:
            await self.db.update_job_status(job.id, JobStatus.APPLIED)

    def run_polling(self) -> None:
        """Run the bot in blocking polling mode (convenience for CLI execution)."""
        print("🤖 Job Hunter Bot is running in polling mode. Press Ctrl+C to stop.")
        self.application.run_polling()


if __name__ == "__main__":
    settings = get_settings()
    database = JobDatabase(settings.database_path)
    bot = JobHunterBot(db=database)
    bot.run_polling()
