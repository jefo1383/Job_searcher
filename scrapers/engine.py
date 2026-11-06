"""Scraping orchestration engine managing collection and deduplication."""

from db.database import (
    JobDatabase,
    generate_job_fingerprints,
    normalize_job_url,
)
from db.models import JobRecord, JobStatus
from scrapers.base import BaseScraper


class ScraperEngine:
    """Orchestrates job collection across multiple scraper sources.

    Attributes:
        db: Initialized JobDatabase persistence manager.
        scrapers: Registry of active BaseScraper instances.
    """

    def __init__(
        self,
        db: JobDatabase,
        scrapers: list[BaseScraper] | None = None,
    ) -> None:
        """Initialize the scraper engine.

        Args:
            db: Target JobDatabase instance.
            scrapers: Optional initial list of scraper instances.
        """
        self.db = db
        self.scrapers: list[BaseScraper] = scrapers or []

    def register_scraper(self, scraper: BaseScraper) -> None:
        """Register a new scraper into the engine.

        Args:
            scraper: BaseScraper instance to register.
        """
        self.scrapers.append(scraper)

    async def collect_jobs(
        self,
        keywords: list[str],
        locations: list[str],
        limit_per_source: int = 20,
    ) -> list[JobRecord]:
        """Execute all registered scrapers and persist non-duplicate jobs.

        Args:
            keywords: Target search terms (job titles, stacks).
            locations: Geographic focus terms.
            limit_per_source: Max items to query per scraper.

        Returns:
            list[JobRecord]: Newly inserted unique job records (status NEW).
        """
        newly_inserted: list[JobRecord] = []

        for scraper in self.scrapers:
            collected_count = 0
            inserted_count = 0

            try:
                # Parcours du flux d'offres généré par le scraper
                async for raw in scraper.scrape(
                    keywords=keywords,
                    locations=locations,
                    limit=limit_per_source,
                ):
                    collected_count += 1
                    # Normalisation stricte de l'URL pour la déduplication
                    norm_url = normalize_job_url(raw.url)

                    # L1: Vérification préliminaire d'URL en base de données
                    existing = await self.db.get_job_by_normalized_url(
                        norm_url
                    )
                    if existing is not None:
                        # Offre déjà connue avec la même URL, on ignore
                        continue

                    # L2: Empreintes sémantiques (strict et tolérant)
                    strict_hash, loose_hash = generate_job_fingerprints(
                        company=raw.company,
                        title=raw.title,
                        location=raw.location,
                    )

                    # Détection d'un doublon multi-sources (Option C)
                    duplicate, rule = await self.db.find_existing_duplicate(
                        content_hash=strict_hash,
                        loose_hash=loose_hash,
                        strict_window_days=30,
                        loose_window_days=14,
                    )
                    if duplicate is not None and duplicate.id is not None:
                        # Offre déjà répertoriée ailleurs : fusion de la source
                        await self.db.add_alternate_url(
                            job_id=duplicate.id,
                            platform=raw.platform,
                            url=raw.url,
                        )
                        print(
                            f"🔗 [Dedup {rule}] Offre '{raw.company}' - "
                            f"'{raw.title}' ({raw.platform}) fusionnée "
                            f"avec #{duplicate.id} ({duplicate.platform})."
                        )
                        continue

                    job_record = JobRecord(
                        platform=raw.platform,
                        company=raw.company,
                        title=raw.title,
                        url=raw.url,
                        normalized_url=norm_url,
                        description=raw.description,
                        raw_payload=raw.raw_payload,
                        content_hash=strict_hash,
                        loose_hash=loose_hash,
                        status=JobStatus.NEW,
                    )

                    saved = await self.db.insert_job(job_record)
                    if saved is not None:
                        newly_inserted.append(saved)
                        inserted_count += 1

                print(
                    f"✓ [{scraper.name}] {collected_count} items analyzed, "
                    f"{inserted_count} new unique jobs inserted."
                )

            except Exception as err:
                # Isolation des erreurs par scraper pour garantir la résilience
                print(f"⚠️ [{scraper.name}] Scraper execution failed: {err}")

        return newly_inserted
