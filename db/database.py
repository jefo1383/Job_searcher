"""Asynchronous SQLite database persistence and deduplication layer."""

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiosqlite

from db.models import JobRecord, JobStatus

# Délais de tolérance pour la déduplication multi-sources
DEFAULT_STRICT_WINDOW_DAYS: int = 30
DEFAULT_LOOSE_WINDOW_DAYS: int = 14

# Suffixes de formes juridiques à retirer pour normaliser l'entreprise
LEGAL_SUFFIXES_REGEX: re.Pattern[str] = re.compile(
    r"\b(sa|sas|sarl|inc|ltd|group|groupe|france|solutions|technologies|corp)\b",
    re.IGNORECASE,
)

# Mentions de contrat et de parité à retirer du titre de poste
JOB_TAGS_REGEX: re.Pattern[str] = re.compile(
    r"(\([fh]\s*/\s*[fh]\)|\b[fh]\s*/\s*[fh]\b|"
    r"\[\s*(cdi|cdd|alternance|stage|interim|freelance)\s*\]|"
    r"\(\s*(cdi|cdd|alternance|stage|interim|freelance)\s*\)|"
    r"\b(cdi|cdd|alternance|stage|interim)\b)",
    re.IGNORECASE,
)

# Alias de localisation cibles vers leur forme canonique
CITY_ALIAS_MAP: dict[str, str] = {
    "marseille": "marseille",
    "aix": "aix",
    "allauch": "allauch",
    "aubagne": "aubagne",
    "marignane": "marignane",
    "istres": "istres",
    "ciotat": "ciotat",
    "vitrolles": "vitrolles",
}

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


def strip_accents(text: str) -> str:
    """Remove diacritical marks from characters for uniform text comparison.

    Args:
        text: Raw text string.

    Returns:
        str: Text with diacritics removed.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_company_name(company: str) -> str:
    """Normalize employer name by stripping legal forms and special characters.

    Args:
        company: Raw company name.

    Returns:
        str: Sanitized lowercase identifier.
    """
    text = strip_accents(company).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = LEGAL_SUFFIXES_REGEX.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_job_title(title: str) -> str:
    """Normalize position title by stripping contract types, gender tags, and symbols.

    Args:
        title: Raw job title string.

    Returns:
        str: Sanitized lowercase role title.
    """
    text = strip_accents(title).lower()
    text = JOB_TAGS_REGEX.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_location(location: str) -> str:
    """Normalize geographical location to a canonical city, department, or remote tag.

    Args:
        location: Raw location string (e.g., 'Marseille (13)', 'Télétravail').

    Returns:
        str: Canonical location token, or empty string if unidentifiable.
    """
    if not location:
        return ""
    text = strip_accents(location).lower()
    if any(k in text for k in ("teletravail", "remote", "distance")):
        return "remote"
    for alias, canonical in CITY_ALIAS_MAP.items():
        if alias in text:
            return canonical
    if "13" in text or "bouches du rhone" in text:
        return "13"
    # Nettoyage générique si aucune ville cible n'est explicitée
    cleaned = re.sub(r"[^\w\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned not in ("france", "paca") else ""


def generate_job_fingerprints(
    company: str,
    title: str,
    location: str = "",
) -> tuple[str | None, str]:
    """Compute strict and loose SHA-256 fingerprints for cross-platform deduplication.

    Args:
        company: Hiring organization name.
        title: Position or role title.
        location: Optional location string.

    Returns:
        tuple[str | None, str]: A pair containing:
            - strict_hash: Hash including location (or None if location is empty).
            - loose_hash: Hash based on company and title only.
    """
    comp = clean_company_name(company)
    tit = clean_job_title(title)
    loc = clean_location(location)

    loose_seed = f"{comp}:{tit}".encode("utf-8")
    loose_hash = hashlib.sha256(loose_seed).hexdigest()

    strict_hash: str | None = None
    if loc:
        strict_seed = f"{comp}:{tit}:{loc}".encode("utf-8")
        strict_hash = hashlib.sha256(strict_seed).hexdigest()

    return strict_hash, loose_hash


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
        """Create tables and indices, ensuring schema migration and WAL mode."""
        # Création du dossier parent si nécessaire
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Activation du mode WAL pour la concurrence en lecture/écriture
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA foreign_keys=ON;")

            # Création initiale de la table des offres d'emploi
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
                    content_hash TEXT,
                    loose_hash TEXT,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # Migration progressive si la table existait sans les colonnes de hash
            cursor = await db.execute("PRAGMA table_info(jobs);")
            columns = [col[1] for col in await cursor.fetchall()]
            if "content_hash" not in columns:
                await db.execute(
                    "ALTER TABLE jobs ADD COLUMN content_hash TEXT;"
                )
            if "loose_hash" not in columns:
                await db.execute(
                    "ALTER TABLE jobs ADD COLUMN loose_hash TEXT;"
                )

            # Index pour accélérer le filtrage par statut et la déduplication
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_norm_url "
                "ON jobs(normalized_url);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_content_hash "
                "ON jobs(content_hash);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_loose_hash "
                "ON jobs(loose_hash);"
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
                        description, raw_payload, score, content_hash,
                        loose_hash, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        job.content_hash,
                        job.loose_hash,
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

    async def find_existing_duplicate(
        self,
        content_hash: str | None,
        loose_hash: str,
        strict_window_days: int = DEFAULT_STRICT_WINDOW_DAYS,
        loose_window_days: int = DEFAULT_LOOSE_WINDOW_DAYS,
    ) -> tuple[JobRecord | None, str | None]:
        """Detect if an incoming job matches an existing posting across platforms.

        Evaluates against a strict fingerprint (with location) within
        strict_window_days (default 30), falling back to a loose fingerprint
        (company + title) within loose_window_days (default 14).

        Args:
            content_hash: Strict fingerprint including location (if available).
            loose_hash: Loose fingerprint of company and title.
            strict_window_days: Max age in days for a strict match (default 30).
            loose_window_days: Max age in days for a loose match (default 14).

        Returns:
            tuple[JobRecord | None, str | None]: The matched record and rule
                name ('strict' or 'loose'), or (None, None) if no duplicate.
        """
        now = datetime.now(timezone.utc)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 1. Vérification stricte : Entreprise + Titre + Localisation
            if content_hash:
                strict_cutoff = (
                    now - timedelta(days=strict_window_days)
                ).isoformat()
                cursor = await db.execute(
                    """
                    SELECT * FROM jobs
                    WHERE content_hash = ? AND created_at >= ?
                    ORDER BY created_at DESC LIMIT 1;
                    """,
                    (content_hash, strict_cutoff),
                )
                row = await cursor.fetchone()
                if row is not None:
                    return self._row_to_record(row), "strict"

            # 2. Vérification tolérante : Entreprise + Titre (< 14 jours)
            loose_cutoff = (
                now - timedelta(days=loose_window_days)
            ).isoformat()
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE loose_hash = ? AND created_at >= ?
                ORDER BY created_at DESC LIMIT 1;
                """,
                (loose_hash, loose_cutoff),
            )
            row = await cursor.fetchone()
            if row is not None:
                return self._row_to_record(row), "loose"

            return None, None

    async def add_alternate_url(
        self,
        job_id: int,
        platform: str,
        url: str,
    ) -> bool:
        """Enrich an existing job record with an alternate portal URL.

        Args:
            job_id: Primary key of target JobRecord.
            platform: Secondary source identifier (e.g. 'indeed', 'hellowork').
            url: Destination URL on the secondary source.

        Returns:
            bool: True if record was found and updated, False otherwise.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT raw_payload, url FROM jobs WHERE id = ?;",
                (job_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False

            raw_payload_text = row["raw_payload"]
            original_url = row["url"]
            payload = (
                json.loads(raw_payload_text)
                if raw_payload_text is not None
                else {}
            )

            # Ne pas dupliquer si c'est déjà l'URL principale
            if original_url == url:
                return False

            alternate_urls: list[dict[str, str]] = payload.setdefault(
                "alternate_urls", []
            )

            # Vérifier si cette URL n'est pas déjà présente
            for entry in alternate_urls:
                if entry.get("url") == url:
                    return False

            now_iso = datetime.now(timezone.utc).isoformat()
            alternate_urls.append(
                {
                    "platform": platform,
                    "url": url,
                    "merged_at": now_iso,
                }
            )

            updated_payload_text = json.dumps(payload)
            update_cursor = await db.execute(
                """
                UPDATE jobs
                SET raw_payload = ?, updated_at = ?
                WHERE id = ?;
                """,
                (updated_payload_text, now_iso, job_id),
            )
            await db.commit()
            return update_cursor.rowcount > 0

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

        content_hash = (
            row["content_hash"] if "content_hash" in row.keys() else None
        )
        loose_hash = (
            row["loose_hash"] if "loose_hash" in row.keys() else None
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
            content_hash=content_hash,
            loose_hash=loose_hash,
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
