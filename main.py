"""Central orchestrator for the Job Hunter Bot autonomous agent."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

from config.criteria import SearchCriteria, parse_search_criteria_file
from config.settings import Settings, get_settings
from core.analyzer import JobAnalyzer
from core.generator import (
    JobGenerator,
    TailoredApplication,
    export_application_markdown,
    sanitize_filename,
)
from core.profile_builder import ProfileBuilder
from db.database import JobDatabase, is_anonymous_company
from db.models import JobStatus
from hitl.telegram_bot import JobHunterBot
from scrapers.bpce import BpceScraper
from scrapers.cma_cgm import CMACGMScraper
from scrapers.engine import ScraperEngine
from scrapers.france_travail import FranceTravailScraper
from scrapers.hellowork import HelloWorkScraper
from scrapers.indeed import IndeedScraper, ReadyCallback
from scrapers.la_bonne_alternance import LaBonneAlternanceScraper
from scrapers.linkedin import LinkedInScraper
from scrapers.orange_cyberdefense import OrangeCyberScraper
from scrapers.thales import ThalesScraper
from scrapers.vinci import VinciScraper
from scrapers.workday import AirbusScraper, FDJScraper
from scrapers.wttj import WTTJScraper

# Mots-clés par défaut alignés sur le profil du candidat
DEFAULT_KEYWORDS: list[str] = [
    "Ingénieur IA",
    "Développeur Backend",
    "Software Engineer",
    "Python",
    "DevOps",
    "DevSecOps",
    "Analyste SOC",
    "CSIRT",
    "Cybersécurité",
    "Systèmes",
]

# Localisations cibles prioritaires
DEFAULT_LOCATIONS: list[str] = [
    "Marseille",
    "Bouches-du-Rhône",
    "13",
    "Aix-en-Provence",
    "Remote",
    "Télétravail",
]


class JobHunterOrchestrator:
    """Master orchestrator combining collection, scoring, and HITL alerts.

    Attributes:
        settings: Application settings instance.
        db: SQLite persistence manager.
        analyzer: Gemini Flash job matching analyzer.
        generator: Application documents and pitch generator.
        bot: Interactive Telegram bot.
        scraper_engine: Multi-source collection engine.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize all subsystems and register scraper connectors.

        Args:
            settings: Optional custom application settings.
        """
        self.settings: Settings = settings or get_settings()

        # 1. Couche de persistance SQLite
        self.db = JobDatabase(self.settings.database_path)

        # 2. Cœur d'analyse et de génération IA
        self.analyzer = JobAnalyzer(
            api_key=self.settings.gemini_api_key.get_secret_value(),
            model_name=self.settings.gemini_model,
            profile_path=self.settings.candidate_profile_path,
        )
        self.generator = JobGenerator(
            api_key=self.settings.gemini_api_key.get_secret_value(),
            model_name=self.settings.gemini_model,
            profile_path=self.settings.candidate_profile_path,
        )

        # 3. Moteur d'analyse du CV et de génération du profil candidat
        self.profile_builder = ProfileBuilder(settings=self.settings)

        # 4. Interface Human-in-the-Loop Telegram
        self.bot = JobHunterBot(
            db=self.db,
            bot_token=self.settings.telegram_bot_token.get_secret_value(),
            chat_id=self.settings.telegram_chat_id,
            profile_builder=self.profile_builder,
            indeed_callback=self._telegram_indeed_runner,
            generator=self.generator,
        )

        # 5. Moteur de scraping multi-sources (Job boards + Portails ATS cibles)
        self.indeed_scraper = IndeedScraper()
        self.scraper_engine = ScraperEngine(db=self.db)
        self.scraper_engine.register_scraper(WTTJScraper())
        self.scraper_engine.register_scraper(FranceTravailScraper())
        self.scraper_engine.register_scraper(LaBonneAlternanceScraper())
        self.scraper_engine.register_scraper(CMACGMScraper())
        self.scraper_engine.register_scraper(HelloWorkScraper())
        self.scraper_engine.register_scraper(self.indeed_scraper)
        self.scraper_engine.register_scraper(LinkedInScraper())
        self.scraper_engine.register_scraper(AirbusScraper())
        self.scraper_engine.register_scraper(ThalesScraper())
        self.scraper_engine.register_scraper(FDJScraper())
        self.scraper_engine.register_scraper(OrangeCyberScraper())
        self.scraper_engine.register_scraper(BpceScraper())
        self.scraper_engine.register_scraper(VinciScraper())

    def _build_active_criteria(
        self,
        keywords: list[str] | None = None,
        locations: list[str] | None = None,
    ) -> SearchCriteria:
        """Load search_criteria.txt and apply optional overrides or defaults."""
        criteria = parse_search_criteria_file(
            self.settings.search_criteria_path
        )
        active_keywords = keywords or criteria.keywords or DEFAULT_KEYWORDS
        active_locations = locations or criteria.locations or DEFAULT_LOCATIONS
        return criteria.model_copy(
            update={
                "keywords": active_keywords,
                "locations": active_locations,
            }
        )

    async def _telegram_indeed_runner(
        self,
        wait_for_ready: ReadyCallback,
    ) -> tuple[int, int]:
        """Run interactive Indeed flow triggered from Telegram /indeed."""
        return await self.run_indeed_interactive(wait_for_ready=wait_for_ready)

    async def run_indeed_interactive(
        self,
        wait_for_ready: ReadyCallback | None = None,
        limit_per_source: int = 20,
    ) -> tuple[int, int]:
        """Launch visible Indeed browser, await user confirmation, and score.

        Args:
            wait_for_ready: Optional async prompt callback (Telegram/CLI).
            limit_per_source: Maximum listings to extract from Indeed.

        Returns:
            tuple[int, int]: Count of newly inserted jobs and qualified alerts.
        """
        active_criteria = self._build_active_criteria()
        interactive_indeed = IndeedScraper(
            headless=False,
            interactive=True,
            wait_for_ready=wait_for_ready,
        )
        new_jobs = await self.scraper_engine._process_scraper(
            scraper=interactive_indeed,
            keywords=active_criteria.keywords,
            locations=active_criteria.locations,
            limit=limit_per_source,
        )
        indeed_ids = [job.id for job in new_jobs if job.id is not None]
        qualified_count = await self._evaluate_and_notify_pending(
            active_criteria,
            job_ids=indeed_ids,
        )
        return len(new_jobs), qualified_count

    async def run_pipeline_cycle(
        self,
        keywords: list[str] | None = None,
        locations: list[str] | None = None,
        limit_per_source: int = 20,
    ) -> int:
        """Execute one complete cycle: scrape, evaluate, pitch, and notify.

        Dynamically loads criteria from search_criteria.txt, falling back to
        provided arguments or application defaults.

        Args:
            keywords: Optional override search terms.
            locations: Optional override geographical filters.
            limit_per_source: Max listings queried per scraper.

        Returns:
            int: Count of newly qualified offers notified via Telegram.
        """
        active_criteria = self._build_active_criteria(
            keywords=keywords,
            locations=locations,
        )

        print("\n🚀 [Cycle] Démarrage de la collecte d'offres d'emploi...")
        new_jobs = await self.scraper_engine.collect_from_criteria(
            criteria=active_criteria,
            limit_per_source=limit_per_source,
        )
        print(f"📦 [Cycle] {len(new_jobs)} nouvelle(s) offre(s) insérée(s).")

        if self.indeed_scraper.last_run_blocked:
            await self.bot.send_info_message(
                "ℹ️ <b>Indeed a bloqué la collecte automatique.</b>\n"
                "Envoyez /indeed ici (ou lancez <code>--indeed</code> en "
                "terminal) lorsque vous êtes devant votre écran pour ouvrir le "
                "navigateur manuellement."
            )

        return await self._evaluate_and_notify_pending(active_criteria)

    async def _evaluate_and_notify_pending(
        self,
        active_criteria: SearchCriteria,
        job_ids: list[int] | None = None,
    ) -> int:
        """Atomically claim NEW offers, score with Gemini, and send alerts."""

        # Réclamation atomique (NEW -> EVALUATING) pour éviter les doublons
        pending_eval_jobs = await self.db.claim_pending_new_jobs(
            limit=100,
            job_ids=job_ids,
        )
        if not pending_eval_jobs:
            print("✨ [Cycle] Aucune offre en attente d'évaluation.")
            return 0

        print(
            f"🧠 [Cycle] Évaluation de {len(pending_eval_jobs)} offre(s) "
            f"par Gemini Flash (Seuil: "
            f"{active_criteria.match_score_threshold}/100, "
            f"Max manquantes: {active_criteria.max_missing_skills_percent}%)..."
        )
        qualified_count = 0

        for job in pending_eval_jobs:
            if job.id is None:
                continue

            try:
                # Évaluation stricte du poste avec seuils configurables
                eval_res, is_qualified = (
                    await self.analyzer.evaluate_and_decide_status(
                        score_threshold=active_criteria.match_score_threshold,
                        title=job.title,
                        company=job.company,
                        description=job.description,
                        platform=job.platform,
                        max_missing_skills_percent=(
                            active_criteria.max_missing_skills_percent
                        ),
                    )
                )

                # Rectification du nom d'entreprise si générique ou masqué
                if eval_res.company and is_anonymous_company(job.company):
                    job.company = eval_res.company.strip()
                if eval_res.contract_type:
                    job.contract_type = eval_res.contract_type.strip()

                payload = dict(job.raw_payload or {})
                payload["evaluation"] = {
                    "score": eval_res.score,
                    "reasoning": eval_res.reasoning,
                    "key_matches": eval_res.key_matches,
                    "missing_skills": eval_res.missing_skills,
                }

                if is_qualified:
                    # Vérification si une lettre existe déjà (en base ou sur disque)
                    folder_name = f"{sanitize_filename(job.company)}_{job.id}"
                    app_file = (
                        Path("Applications") / folder_name / "lettre_motivation.md"
                    )
                    existing_app = payload.get("application")

                    has_payload_letter = (
                        isinstance(existing_app, dict)
                        and bool(existing_app.get("full_cover_letter"))
                    )

                    app_doc: TailoredApplication | None = None
                    if has_payload_letter and isinstance(existing_app, dict):
                        print(
                            f"  📄 [Application] Lettre existante conservée "
                            f"en base pour #{job.id} ({job.company})."
                        )
                        app_doc = TailoredApplication(
                            language=str(existing_app.get("language") or "fr"),
                            subject_line=str(
                                existing_app.get("subject_line")
                                or f"Candidature - {job.title}"
                            ),
                            quick_pitch=str(
                                existing_app.get("quick_pitch") or ""
                            ),
                            full_cover_letter=str(
                                existing_app["full_cover_letter"]
                            ),
                        )
                    elif app_file.exists():
                        print(
                            f"  📄 [Application] Lettre existante réutilisée "
                            f"depuis {app_file} pour #{job.id}."
                        )
                        disk_text = app_file.read_text(encoding="utf-8")
                        extracted_pitch = ""
                        extracted_letter = ""
                        if "## 📌 Pitch d'accroche" in disk_text:
                            p_part = disk_text.split(
                                "## 📌 Pitch d'accroche", 1
                            )[1]
                            extracted_pitch = p_part.split("---", 1)[0].strip()
                        if "## 📄 Lettre de motivation personnalisée" in disk_text:
                            l_part = disk_text.split(
                                "## 📄 Lettre de motivation personnalisée", 1
                            )[1]
                            extracted_letter = l_part.split("---", 1)[0].strip()

                        app_doc = TailoredApplication(
                            language="fr",
                            subject_line=f"Candidature - {job.title}",
                            quick_pitch=extracted_pitch or "Candidature ciblée",
                            full_cover_letter=extracted_letter or disk_text,
                        )
                    else:
                        # Vérification d'un dossier pour la même entreprise et titre
                        company_slug = sanitize_filename(job.company)
                        app_root = Path("Applications")
                        if app_root.exists() and company_slug:
                            for cand_dir in app_root.glob(f"{company_slug}_*"):
                                cand_md = cand_dir / "lettre_motivation.md"
                                if cand_md.exists():
                                    try:
                                        cand_text = cand_md.read_text(
                                            encoding="utf-8"
                                        )
                                        if job.title.lower() in cand_text.lower():
                                            print(
                                                f"  🔗 [Application] Lettre "
                                                f"similaire réutilisée depuis "
                                                f"{cand_md} pour #{job.id}."
                                            )
                                            p_m = ""
                                            l_m = ""
                                            if "## 📌 Pitch d'accroche" in cand_text:
                                                p_sub = cand_text.split(
                                                    "## 📌 Pitch d'accroche", 1
                                                )[1]
                                                p_m = p_sub.split("---", 1)[0].strip()
                                            letter_marker = (
                                                "## 📄 Lettre de motivation "
                                                "personnalisée"
                                            )
                                            if letter_marker in cand_text:
                                                l_sub = cand_text.split(
                                                    letter_marker, 1
                                                )[1]
                                                l_m = l_sub.split(
                                                    "---", 1
                                                )[0].strip()
                                            app_doc = TailoredApplication(
                                                language="fr",
                                                subject_line=(
                                                    f"Candidature - {job.title}"
                                                ),
                                                quick_pitch=(
                                                    p_m or "Candidature ciblée"
                                                ),
                                                full_cover_letter=(
                                                    l_m or cand_text
                                                ),
                                            )
                                            break
                                    except Exception:
                                        pass

                    # Si aucune lettre existante n'a été trouvée, génération Gemini
                    if app_doc is None:
                        print(
                            f"  ✍️ [Application] Génération d'une nouvelle "
                            f"lettre pour #{job.id} ({job.company})..."
                        )
                        app_doc = await self.generator.generate_application(
                            title=job.title,
                            company=job.company,
                            description=job.description,
                            key_matches=eval_res.key_matches,
                            missing_skills=eval_res.missing_skills,
                            platform=job.platform,
                        )

                    # Export Markdown persistant dans Applications/ (sans écraser)
                    export_application_markdown(
                        job_id=job.id,
                        company=job.company,
                        title=job.title,
                        url=job.url,
                        score=eval_res.score,
                        contract_type=job.contract_type,
                        application=app_doc,
                        key_matches=eval_res.key_matches,
                        missing_skills=eval_res.missing_skills,
                        reasoning=eval_res.reasoning,
                        overwrite=False,
                    )

                    payload["application"] = {
                        "language": app_doc.language,
                        "subject_line": app_doc.subject_line,
                        "quick_pitch": app_doc.quick_pitch,
                        "full_cover_letter": app_doc.full_cover_letter,
                    }
                    job.raw_payload = payload

                    # Mise à jour du statut, de l'entreprise et du contrat en base
                    await self.db.update_job_evaluation(
                        job_id=job.id,
                        score=eval_res.score,
                        status=JobStatus.PENDING_REVIEW,
                        company=job.company,
                        contract_type=job.contract_type,
                        raw_payload=payload,
                    )

                    # Verrou atomique pour garantir une notification Telegram unique
                    notified = False
                    if await self.db.mark_job_notified(job.id):
                        notified = await self.bot.send_job_alert(
                            job=job,
                            analysis=eval_res,
                            pitch=app_doc.quick_pitch,
                        )
                        if not notified:
                            await self.db.clear_job_notified(job.id)
                    status_log = (
                        "notifiée sur Telegram" if notified else "sauvegardée"
                    )
                    print(
                        f"  🎯 Qualifiée ({eval_res.score}/100) : "
                        f"{job.company} - {job.title} [{status_log}]"
                    )
                    qualified_count += 1
                else:
                    # Offre rejetée (note insuffisante ou critères éliminatoires)
                    job.raw_payload = payload
                    await self.db.update_job_evaluation(
                        job_id=job.id,
                        score=eval_res.score,
                        status=JobStatus.REJECTED,
                        company=(
                            eval_res.company
                            if eval_res.company
                            and is_anonymous_company(job.company)
                            else None
                        ),
                        contract_type=job.contract_type,
                        raw_payload=payload,
                    )
                    print(
                        f"  ❌ Rejetée ({eval_res.score}/100) : "
                        f"{job.company} - {job.title} ({eval_res.reasoning[:60]}...)"
                    )

            except Exception as err:
                await self.db.release_claimed_job(job.id)
                print(
                    f"  ⚠️ Erreur lors de l'évaluation de l'offre #{job.id} "
                    f"({job.company}) : {err}"
                )

        return qualified_count

    async def run_periodic_watcher(
        self,
        interval_hours: float,
        keywords: list[str] | None = None,
        locations: list[str] | None = None,
        limit_per_source: int = 20,
    ) -> None:
        """Run periodic watch loop with scheduled sleep intervals.

        Args:
            interval_hours: Delay in hours between complete scraping cycles.
            keywords: Target keywords.
            locations: Geographical targets.
            limit_per_source: Max items per source.
        """
        sleep_seconds = interval_hours * 3600.0

        while True:
            try:
                await self.run_pipeline_cycle(
                    keywords=keywords,
                    locations=locations,
                    limit_per_source=limit_per_source,
                )
            except Exception as err:
                print(f"⚠️ [Watcher] Erreur inattendue dans le cycle : {err}")

            print(
                f"\n💤 [Watcher] Mise en sommeil pour {interval_hours:.1f}h. "
                "Le bot Telegram reste actif et à l'écoute sur mobile."
            )
            await asyncio.sleep(sleep_seconds)

    async def start(
        self,
        once: bool = False,
        bot_only: bool = False,
        indeed: bool = False,
        update_profile: bool = False,
        interval_hours: float = 168.0,
        keywords: list[str] | None = None,
        locations: list[str] | None = None,
        limit_per_source: int = 20,
    ) -> None:
        """Start the orchestrator in requested operating mode.

        Args:
            once: If True, execute one collection cycle and exit.
            bot_only: If True, only launch interactive Telegram bot.
            indeed: If True, launch visible Chrome for manual Indeed scrape.
            update_profile: If True, force profile.md regeneration from CV.
            interval_hours: Hours between automated scraping cycles.
            keywords: Custom search keywords list.
            locations: Custom geographical locations list.
            limit_per_source: Max results per scraper.
        """
        # 1. Vérification et synchronisation automatique du profil candidat
        profile_regenerated = await self.profile_builder.sync_profile(
            force=update_profile
        )
        if profile_regenerated:
            # Rechargement de l'analyseur et du générateur avec le profil frais
            self.analyzer = JobAnalyzer(
                api_key=self.settings.gemini_api_key.get_secret_value(),
                model_name=self.settings.gemini_model,
                profile_path=self.settings.candidate_profile_path,
            )
            self.generator = JobGenerator(
                api_key=self.settings.gemini_api_key.get_secret_value(),
                model_name=self.settings.gemini_model,
                profile_path=self.settings.candidate_profile_path,
            )
            self.bot.generator = self.generator

        # 2. Initialisation de la persistance SQLite
        await self.db.init_db()

        target_keywords = keywords or DEFAULT_KEYWORDS
        target_locations = locations or DEFAULT_LOCATIONS

        # 3. Initialisation de l'application Telegram
        telegram_app = self.bot.application
        await telegram_app.initialize()
        await telegram_app.start()

        if indeed:
            # Mode interactif dédié Indeed avec navigateur visible
            try:
                collected, qualified = await self.run_indeed_interactive(
                    limit_per_source=limit_per_source
                )
                print(
                    f"🏁 [Indeed] {collected} offre(s) collectée(s), "
                    f"{qualified} qualifiée(s)."
                )
            finally:
                await telegram_app.stop()
                await telegram_app.shutdown()
            return

        if once:
            # Mode One-Shot : exécution immédiate sans boucle persistante
            try:
                await self.run_pipeline_cycle(
                    keywords=target_keywords,
                    locations=target_locations,
                    limit_per_source=limit_per_source,
                )
            finally:
                await telegram_app.stop()
                await telegram_app.shutdown()
            print("🏁 [Orchestrator] Cycle one-shot terminé.")
            return

        updater = telegram_app.updater
        if updater is not None:
            await updater.start_polling()

        print("🤖 [Orchestrator] Bot Telegram en écoute active (Long Polling).")

        watcher_task: asyncio.Task[Any] | None = None
        if not bot_only:
            # Lancement de la tâche périodique conjointe
            watcher_task = asyncio.create_task(
                self.run_periodic_watcher(
                    interval_hours=interval_hours,
                    keywords=target_keywords,
                    locations=target_locations,
                    limit_per_source=limit_per_source,
                )
            )

        try:
            # Maintien de l'event loop actif
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            pass
        finally:
            print("\n🛑 [Orchestrator] Arrêt propre de l'agent en cours...")
            if watcher_task is not None:
                watcher_task.cancel()
            if updater is not None:
                await updater.stop()
            await telegram_app.stop()
            await telegram_app.shutdown()
            print("👋 [Orchestrator] Agent arrêté proprement.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for the agent.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="job-hunter",
        description="Autonomous Job Hunter Bot with Gemini Flash and Telegram HITL.",
    )
    parser.add_argument(
        "--once",
        "--now",
        action="store_true",
        dest="once",
        help="Execute a single scraping & evaluation cycle and exit immediately.",
    )
    parser.add_argument(
        "--bot-only",
        action="store_true",
        dest="bot_only",
        help="Only launch Telegram bot to answer commands without scraping.",
    )
    parser.add_argument(
        "--indeed",
        action="store_true",
        dest="indeed",
        help="Open visible browser for manual Indeed loading, then scrape.",
    )
    parser.add_argument(
        "--update-profile",
        action="store_true",
        dest="update_profile",
        help="Force profile.md regeneration from the latest CV in Profile/.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=168.0,
        dest="interval_hours",
        help="Scheduled hours between scraping passes (default: 168.0 = 1 week).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        dest="limit",
        help="Maximum listings fetched per scraper (default: 20).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint compatible with cross-platform packaging (PyInstaller).

    Args:
        argv: Optional command-line arguments sequence.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    orchestrator = JobHunterOrchestrator()

    try:
        asyncio.run(
            orchestrator.start(
                once=args.once,
                bot_only=args.bot_only,
                indeed=args.indeed,
                update_profile=args.update_profile,
                interval_hours=args.interval_hours,
                limit_per_source=args.limit,
            )
        )
    except KeyboardInterrupt:
        print("\n👋 Interruption utilisateur (Ctrl+C). Fin de session.")
        sys.exit(0)


if __name__ == "__main__":
    main()
