"""Central orchestrator for the Job Hunter Bot autonomous agent."""

import argparse
import asyncio
from collections.abc import Sequence
import sys
from typing import Any

from config.settings import Settings, get_settings
from core.analyzer import JobAnalyzer
from core.generator import JobGenerator
from db.database import JobDatabase
from db.models import JobStatus
from hitl.telegram_bot import JobHunterBot
from scrapers.cma_cgm import CMACGMScraper
from scrapers.engine import ScraperEngine
from scrapers.france_travail import FranceTravailScraper
from scrapers.hellowork import HelloWorkScraper
from scrapers.indeed import IndeedScraper
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

        # 3. Interface Human-in-the-Loop Telegram
        self.bot = JobHunterBot(
            db=self.db,
            bot_token=self.settings.telegram_bot_token.get_secret_value(),
            chat_id=self.settings.telegram_chat_id,
        )

        # 4. Moteur de scraping multi-sources
        self.scraper_engine = ScraperEngine(db=self.db)
        self.scraper_engine.register_scraper(WTTJScraper())
        self.scraper_engine.register_scraper(FranceTravailScraper())
        self.scraper_engine.register_scraper(CMACGMScraper())
        self.scraper_engine.register_scraper(HelloWorkScraper())
        self.scraper_engine.register_scraper(IndeedScraper())

    async def run_pipeline_cycle(
        self,
        keywords: list[str],
        locations: list[str],
        limit_per_source: int = 20,
    ) -> int:
        """Execute one complete cycle: scrape, evaluate, pitch, and notify.

        Args:
            keywords: Target search terms.
            locations: Geographical filters.
            limit_per_source: Max listings queried per scraper.

        Returns:
            int: Count of newly qualified offers notified via Telegram.
        """
        print("\n🚀 [Cycle] Démarrage de la collecte d'offres d'emploi...")
        new_jobs = await self.scraper_engine.collect_jobs(
            keywords=keywords,
            locations=locations,
            limit_per_source=limit_per_source,
        )
        print(f"📦 [Cycle] {len(new_jobs)} nouvelle(s) offre(s) insérée(s).")

        # Récupération des offres en attente d'évaluation (statut NEW)
        pending_eval_jobs = await self.db.get_jobs_by_status(
            JobStatus.NEW, limit=100
        )
        if not pending_eval_jobs:
            print("✨ [Cycle] Aucune offre en attente d'évaluation.")
            return 0

        print(
            f"🧠 [Cycle] Évaluation de {len(pending_eval_jobs)} offre(s) "
            f"par Gemini Flash..."
        )
        qualified_count = 0

        for job in pending_eval_jobs:
            if job.id is None:
                continue

            try:
                # Évaluation stricte du poste par rapport au profil candidat
                eval_res, is_qualified = (
                    await self.analyzer.evaluate_and_decide_status(
                        score_threshold=self.settings.match_score_threshold,
                        title=job.title,
                        company=job.company,
                        description=job.description,
                        platform=job.platform,
                    )
                )

                if is_qualified:
                    # Génération du pitch ciblé et de la lettre pour Telegram/ATS
                    app_doc = await self.generator.generate_application(
                        title=job.title,
                        company=job.company,
                        description=job.description,
                        key_matches=eval_res.key_matches,
                        missing_skills=eval_res.missing_skills,
                        platform=job.platform,
                    )

                    # Mise à jour du statut en base de données
                    await self.db.update_job_evaluation(
                        job_id=job.id,
                        score=eval_res.score,
                        status=JobStatus.PENDING_REVIEW,
                    )

                    # Envoi de la notification interactive sur Telegram
                    notified = await self.bot.send_job_alert(
                        job=job,
                        analysis=eval_res,
                        pitch=app_doc.quick_pitch,
                    )
                    status_log = "notifiée sur Telegram" if notified else "sauvegardée"
                    print(
                        f"  🎯 Qualifiée ({eval_res.score}/100) : "
                        f"{job.company} - {job.title} [{status_log}]"
                    )
                    qualified_count += 1
                else:
                    # Offre rejetée (note insuffisante ou critères éliminatoires)
                    await self.db.update_job_evaluation(
                        job_id=job.id,
                        score=eval_res.score,
                        status=JobStatus.REJECTED,
                    )
                    print(
                        f"  ❌ Rejetée ({eval_res.score}/100) : "
                        f"{job.company} - {job.title} ({eval_res.reasoning[:60]}...)"
                    )

            except Exception as err:
                print(
                    f"  ⚠️ Erreur lors de l'évaluation de l'offre #{job.id} "
                    f"({job.company}) : {err}"
                )

        return qualified_count

    async def run_periodic_watcher(
        self,
        interval_hours: float,
        keywords: list[str],
        locations: list[str],
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
        interval_hours: float = 168.0,
        keywords: list[str] | None = None,
        locations: list[str] | None = None,
        limit_per_source: int = 20,
    ) -> None:
        """Start the orchestrator in requested operating mode.

        Args:
            once: If True, execute one collection cycle and exit.
            bot_only: If True, only launch interactive Telegram bot.
            interval_hours: Hours between automated scraping cycles.
            keywords: Custom search keywords list.
            locations: Custom geographical locations list.
            limit_per_source: Max results per scraper.
        """
        # Initialisation de la persistance SQLite
        await self.db.init_db()

        target_keywords = keywords or DEFAULT_KEYWORDS
        target_locations = locations or DEFAULT_LOCATIONS

        if once:
            # Mode One-Shot : exécution immédiate sans boucle ni bot persistant
            await self.run_pipeline_cycle(
                keywords=target_keywords,
                locations=target_locations,
                limit_per_source=limit_per_source,
            )
            print("🏁 [Orchestrator] Cycle one-shot terminé.")
            return

        # Démarrage de l'écoute Telegram en arrière-plan
        telegram_app = self.bot.application
        await telegram_app.initialize()
        await telegram_app.start()
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
                interval_hours=args.interval_hours,
                limit_per_source=args.limit,
            )
        )
    except KeyboardInterrupt:
        print("\n👋 Interruption utilisateur (Ctrl+C). Fin de session.")
        sys.exit(0)


if __name__ == "__main__":
    main()
