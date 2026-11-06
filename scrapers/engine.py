"""Scraping orchestration engine managing collection and deduplication."""

from urllib.parse import urlparse

from config.criteria import SearchCriteria
from db.database import (
    JobDatabase,
    generate_job_fingerprints,
    normalize_job_url,
)
from db.models import JobRecord, JobStatus
from scrapers.base import BaseScraper
from scrapers.generic_browser import GenericBrowserScraper


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
            inserted = await self._process_scraper(
                scraper=scraper,
                keywords=keywords,
                locations=locations,
                limit=limit_per_source,
            )
            newly_inserted.extend(inserted)
        return newly_inserted

    async def collect_from_url(
        self,
        url: str,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> list[JobRecord]:
        """Scrape jobs from a specific target URL using dedicated or fallback scraper.

        Args:
            url: Target jobboard or career site URL.
            keywords: Job search keywords.
            locations: Geographic filters.
            limit: Maximum items to extract.

        Returns:
            list[JobRecord]: Newly inserted unique job records.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Recherche si un scraper dédié correspond au domaine de l'URL
        target_scraper: BaseScraper | None = None
        for s in self.scrapers:
            if s.name.lower() in domain or domain in s.name.lower():
                target_scraper = s
                break

        # Fallback automatique sur le scraper universel Playwright (Tier 2)
        if target_scraper is None:
            print(
                f"🌐 [Router] Pas de connecteur dédié pour '{domain}'. "
                f"Lancement du fallback universel Playwright..."
            )
            target_scraper = GenericBrowserScraper(target_url=url)

        return await self._process_scraper(
            scraper=target_scraper,
            keywords=keywords,
            locations=locations,
            limit=limit,
        )

    async def collect_from_criteria(
        self,
        criteria: SearchCriteria,
        limit_per_source: int = 20,
    ) -> list[JobRecord]:
        """Collect postings based on human-configured SearchCriteria.

        Executes all registered scrapers and visits any custom URLs specified
        in the criteria configuration.

        Args:
            criteria: Validated SearchCriteria model instance.
            limit_per_source: Max items per scraper source.

        Returns:
            list[JobRecord]: All newly persisted unique jobs.
        """
        all_inserted: list[JobRecord] = []

        # 1. Collecte standard via les connecteurs enregistrés
        standard_inserted = await self.collect_jobs(
            keywords=criteria.keywords,
            locations=criteria.locations,
            limit_per_source=limit_per_source,
        )
        all_inserted.extend(standard_inserted)

        # 2. Collecte ciblée sur les URLs personnalisées indiquées
        for custom_url in criteria.custom_urls:
            try:
                url_inserted = await self.collect_from_url(
                    url=custom_url,
                    keywords=criteria.keywords,
                    locations=criteria.locations,
                    limit=limit_per_source,
                )
                all_inserted.extend(url_inserted)
            except Exception as err:
                print(
                    f"⚠️ [Router] Erreur lors de l'inspection de "
                    f"'{custom_url}': {err}"
                )

        return all_inserted

    async def _process_scraper(
        self,
        scraper: BaseScraper,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> list[JobRecord]:
        """Stream items from a scraper, applying URL and multi-source deduplication.

        Args:
            scraper: Target BaseScraper instance.
            keywords: Keywords to query.
            locations: Geographic criteria.
            limit: Maximum items to query.

        Returns:
            list[JobRecord]: List of newly inserted jobs.
        """
        newly_inserted: list[JobRecord] = []
        collected_count = 0
        inserted_count = 0

        try:
            async for raw in scraper.scrape(
                keywords=keywords,
                locations=locations,
                limit=limit,
            ):
                collected_count += 1
                norm_url = normalize_job_url(raw.url)

                # L1: Vérification préliminaire d'URL en base de données
                existing = await self.db.get_job_by_normalized_url(norm_url)
                if existing is not None:
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
            print(f"⚠️ [{scraper.name}] Scraper execution failed: {err}")

        return newly_inserted
