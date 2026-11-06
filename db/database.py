"""Asynchronous SQLite database persistence and deduplication layer."""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiosqlite

from db.models import JobRecord, JobStatus

# Paramètres de tracking à retirer pour la normalisation d'URL
DEFAULT_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "refid",
        "trackingid",
        "ref",
        "trk",
        "trkinfo",
        "midtoken",
        "from",
        "fromage",
        "tk",
        "vjs",
        "spm",
        "fbclid",
        "gclid",
        "msclkid",
        "yclid",
    }
)


def normalize_job_url(
    raw_url: str,
    params_to_strip: frozenset[str] = DEFAULT_TRACKING_PARAMS,
) -> str:
    """Sanitize and normalize a job offer URL to ensure reliable deduplication.

    Strips analytics/tracking query parameters, removes fragments,
    lowercases scheme and host, sorts remaining query parameters, and trims
    trailing slashes.

    Args:
        raw_url: Scraped job offer URL.
        params_to_strip: Set of query parameter keys to filter out.

    Returns:
        str: Canonicalized URL for deduplication.
    """
    # Décomposition de l'URL
    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Nettoyage du chemin : suppression des slashes finaux superflus
    path = parsed.path.rstrip("/")
    if not path and scheme:
        path = ""

    # Filtrage et tri déterministe des paramètres de requête
    filtered_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in params_to_strip
    ]
    sorted_params = sorted(filtered_params, key=lambda pair: pair[0])
    query = urlencode(sorted_params)

    # Reconstruction de l'URL sans fragment
    return urlunparse((scheme, netloc, path, "", query, ""))


class JobDatabase:
    """Async SQLite persistence manager for job records and status tracking.

    Attributes:
        db_path: Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the database manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path

    async def init_db(self) -> None:
        """Create tables and indices, ensuring WAL mode is enabled."""
        # Création du dossier parent si nécessaire
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Activation du mode WAL pour la concurrence en lecture/écriture
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA foreign_keys=ON;")

            # Création de la table des offres d'emploi
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    normalized_url TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    raw_payload TEXT,
                    score INTEGER,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # Index pour accélérer le filtrage par statut et la déduplication
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_norm_url "
                "ON jobs(normalized_url);"
            )
            await db.commit()

    async def insert_job(self, job: JobRecord) -> JobRecord | None:
        """Insert a newly scraped job offer with strict deduplication.

        Args:
            job: Populated JobRecord instance to persist.

        Returns:
            JobRecord | None: Saved JobRecord with populated id, or None if
                a job with the same normalized_url already exists.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        payload_json = (
            json.dumps(job.raw_payload) if job.raw_payload is not None else None
        )

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                # Insertion protégée par la contrainte UNIQUE sur normalized_url
                cursor = await db.execute(
                    """
                    INSERT INTO jobs (
                        platform, company, title, url, normalized_url,
                        description, raw_payload, score, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.platform,
                        job.company,
                        job.title,
                        job.url,
                        job.normalized_url,
                        job.description,
                        payload_json,
                        job.score,
                        job.status.value,
                        now_iso,
                        now_iso,
                    ),
                )
                await db.commit()
                # Récupération de la clé primaire générée
                generated_id = cursor.lastrowid
                return job.model_copy(
                    update={
                        "id": generated_id,
                        "created_at": datetime.fromisoformat(now_iso),
                        "updated_at": datetime.fromisoformat(now_iso),
                    }
                )
            except aiosqlite.IntegrityError:
                # Doublon détecté via la contrainte UNIQUE
                return None

    async def get_job_by_id(self, job_id: int) -> JobRecord | None:
        """Fetch a job record by its primary key identifier.

        Args:
            job_id: SQLite primary key.

        Returns:
            JobRecord | None: Deserialized record, or None if not found.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE id = ?;",
                (job_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    async def get_job_by_normalized_url(
        self, normalized_url: str
    ) -> JobRecord | None:
        """Fetch a job record by its canonicalized URL.

        Args:
            normalized_url: Sanitized unique URL.

        Returns:
            JobRecord | None: Deserialized record, or None if not found.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE normalized_url = ?;",
                (normalized_url,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    async def get_jobs_by_status(
        self,
        status: JobStatus,
        limit: int = 50,
    ) -> list[JobRecord]:
        """Fetch job records filtered by processing lifecycle status.

        Args:
            status: Target JobStatus enum value.
            limit: Maximum count of rows to return.

        Returns:
            list[JobRecord]: List of matching job records.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY id ASC LIMIT ?;",
                (status.value, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    async def update_job_evaluation(
        self,
        job_id: int,
        score: int,
        status: JobStatus,
    ) -> bool:
        """Update job score and status after LLM evaluation.

        Args:
            job_id: SQLite primary key.
            score: Computed match score (0-100).
            status: New status (e.g., PENDING_REVIEW or REJECTED).

        Returns:
            bool: True if a row was updated, False otherwise.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET score = ?, status = ?, updated_at = ?
                WHERE id = ?;
                """,
                (score, status.value, now_iso, job_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def update_job_status(self, job_id: int, status: JobStatus) -> bool:
        """Transition a job record to a new lifecycle status.

        Args:
            job_id: SQLite primary key.
            status: Destination JobStatus.

        Returns:
            bool: True if a row was updated, False otherwise.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?;",
                (status.value, now_iso, job_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def count_jobs_by_status(self) -> dict[JobStatus, int]:
        """Aggregate counts of stored jobs grouped by their lifecycle status.

        Returns:
            dict[JobStatus, int]: Mapping from each status to its count.
        """
        counts: dict[JobStatus, int] = {status: 0 for status in JobStatus}
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status;"
            )
            rows = await cursor.fetchall()
            for status_str, count in rows:
                try:
                    counts[JobStatus(status_str)] = count
                except ValueError:
                    continue
        return counts

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> JobRecord:
        """Convert a database row mapping into a typed JobRecord.

        Args:
            row: SQLite row with column access.

        Returns:
            JobRecord: Deserialized Pydantic instance.
        """
        raw_payload_text = row["raw_payload"]
        raw_payload = (
            json.loads(raw_payload_text)
            if raw_payload_text is not None
            else None
        )

        return JobRecord(
            id=row["id"],
            platform=row["platform"],
            company=row["company"],
            title=row["title"],
            url=row["url"],
            normalized_url=row["normalized_url"],
            description=row["description"],
            raw_payload=raw_payload,
            score=row["score"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
