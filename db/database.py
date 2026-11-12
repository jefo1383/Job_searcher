"""Asynchronous SQLite database persistence and deduplication layer."""

from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiosqlite

from db.models import JobRecord, JobStatus

# Délais de tolérance pour la déduplication multi-sources
DEFAULT_STRICT_WINDOW_DAYS: int = 30
DEFAULT_LOOSE_WINDOW_DAYS: int = 14

# Délai avant libération d'un verrou d'évaluation bloqué (reprise après crash)
DEFAULT_STALE_EVALUATING_MINUTES: int = 10

# Seuils réglables pour la déduplication B1 et la ré-évaluation sélective
DEFAULT_MIN_SPECIFIC_TITLE_WORDS: int = 4
DEFAULT_MIN_SPECIFIC_TITLE_CHARS: int = 25
DEFAULT_CROSS_COMPANY_SIMILARITY: float = 0.65
DEFAULT_STUB_DESCRIPTION_LENGTH: int = 450
DEFAULT_RICHER_DESCRIPTION_RATIO: float = 1.8
DEFAULT_RICHER_DESCRIPTION_DELTA: int = 400
DEFAULT_SHORT_DESC_THRESHOLD: int = 300
DEFAULT_SHORT_DESC_MIN_UPGRADE: int = 500
DEFAULT_SQLITE_TIMEOUT_SECONDS: float = 15.0

# Suffixes de formes juridiques à retirer pour normaliser l'entreprise
LEGAL_SUFFIXES_REGEX: re.Pattern[str] = re.compile(
    r"\b(sa|sas|sarl|inc|ltd|group|groupe|france|solutions|technologies|corp)\b",
    re.IGNORECASE,
)

# Mentions de contrat et de parité à retirer du titre de poste
JOB_TAGS_REGEX: re.Pattern[str] = re.compile(
    r"([\(\[]\s*(?:all\s+genders?|tous\s+genres|"
    r"[xhfm](?:\s*/\s*[xhfm]){1,2})\s*[\)\]]|"
    r"\b[xhfm]\s*/\s*[xhfm](?:\s*/\s*[xhfm])?\b|"
    r"[\(\[]\s*(?:cdi|cdd|alternance|stage|interim|internship|freelance)"
    r"\s*[\)\]]|"
    r"\b(cdi|cdd|alternance|stage|interim|internship|freelance)\b)",
    re.IGNORECASE,
)

# Noms d'entreprises génériques ou masqués (France Travail, agrégateurs)
ANONYMOUS_COMPANY_NAMES: frozenset[str] = frozenset(
    {
        "enseigne non disponible",
        "entreprise non communiquee",
        "entreprise non disponible",
        "confidentiel",
        "entreprise confidentielle",
        "anonyme",
        "non communique",
        "non communiquee",
        "non precise",
        "non precisee",
        "unknown",
        "inconnue",
        "inconnu",
        "na",
        "n a",
        "aucune",
        "hellowork",
        "listing",
    }
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
    # Décomposition de l'URL et normalisation du sous-domaine www.
    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

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


def is_anonymous_company(company: str) -> bool:
    """Return True when the employer name is empty or an anonymized placeholder.

    Args:
        company: Raw employer name string.

    Returns:
        bool: True if company cannot reliably identify a real employer.
    """
    cleaned = clean_company_name(company)
    if not cleaned:
        return True
    return cleaned in ANONYMOUS_COMPANY_NAMES


def clean_job_title(title: str, company: str = "") -> str:
    """Normalize position title by stripping contract types, gender tags, and symbols.

    Args:
        title: Raw job title string.
        company: Optional employer name stripped when appended to title.

    Returns:
        str: Sanitized lowercase role title.
    """
    text = strip_accents(title).lower()
    text = JOB_TAGS_REGEX.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    normalized = re.sub(r"\s+", " ", text).strip()

    # Suppression d'un éventuel nom d'entreprise accolé en fin de titre (HelloWork)
    if company and not is_anonymous_company(company):
        comp = clean_company_name(company)
        if comp and normalized.endswith(f" {comp}"):
            normalized = normalized[: -len(comp)].strip()

    return normalized


def clean_location(location: str, fallback_title: str = "") -> str:
    """Normalize geographical location to a canonical city, department, or remote tag.

    Args:
        location: Raw location string (e.g., 'Marseille (13)', 'Télétravail').
        fallback_title: Optional job title scanned if location is missing.

    Returns:
        str: Canonical location token, or empty string if unidentifiable.
    """
    if location:
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
        if cleaned and cleaned not in ("france", "paca"):
            return cleaned

    # Repli : détection de la ville directement dans le titre si fournie
    if fallback_title:
        title_text = strip_accents(fallback_title).lower()
        for alias, canonical in CITY_ALIAS_MAP.items():
            if re.search(rf"\b{re.escape(alias)}\b", title_text):
                return canonical

    return ""


def generate_title_loc_hash(
    title: str,
    location: str = "",
    company: str = "",
) -> str | None:
    """Compute SHA-256 fingerprint from normalized title and canonical location.

    Args:
        title: Position or role title.
        location: Optional location string.
        company: Optional company name to strip trailing brand repetitions.

    Returns:
        str | None: SHA-256 hex digest when both title and location resolve.
    """
    tit = clean_job_title(title, company=company)
    loc = clean_location(location, fallback_title=title)
    if not tit or not loc:
        return None
    seed = f"{tit}:{loc}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()


def generate_all_fingerprints(
    company: str,
    title: str,
    location: str = "",
) -> tuple[str | None, str, str | None]:
    """Compute strict, loose, and title-location SHA-256 fingerprints.

    Args:
        company: Hiring organization name.
        title: Position or role title.
        location: Optional location string.

    Returns:
        tuple[str | None, str, str | None]: Strict hash, loose hash, and
            title+location hash.
    """
    comp = clean_company_name(company)
    tit = clean_job_title(title, company=company)
    loc = clean_location(location, fallback_title=title)

    loose_seed = f"{comp}:{tit}".encode("utf-8")
    loose_hash = hashlib.sha256(loose_seed).hexdigest()

    strict_hash: str | None = None
    if loc:
        strict_seed = f"{comp}:{tit}:{loc}".encode("utf-8")
        strict_hash = hashlib.sha256(strict_seed).hexdigest()

    title_loc_hash = generate_title_loc_hash(
        title=title,
        location=location,
        company=company,
    )
    return strict_hash, loose_hash, title_loc_hash


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
        tuple[str | None, str]: Strict hash (with location) and loose hash.
    """
    strict_hash, loose_hash, _ = generate_all_fingerprints(
        company=company,
        title=title,
        location=location,
    )
    return strict_hash, loose_hash


def compute_description_similarity(desc_a: str, desc_b: str) -> float:
    """Compute normalized text similarity ratio between two descriptions.

    Args:
        desc_a: First job description.
        desc_b: Second job description.

    Returns:
        float: Similarity score in [0.0, 1.0].
    """
    norm_a = re.sub(r"\s+", " ", strip_accents(desc_a).lower()).strip()[:1500]
    norm_b = re.sub(r"\s+", " ", strip_accents(desc_b).lower()).strip()[:1500]
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def is_description_significantly_richer(
    existing_description: str,
    candidate_description: str,
    min_ratio: float = DEFAULT_RICHER_DESCRIPTION_RATIO,
    min_delta: int = DEFAULT_RICHER_DESCRIPTION_DELTA,
) -> bool:
    """Determine whether a duplicate description warrants re-evaluation.

    Args:
        existing_description: Description currently stored in SQLite.
        candidate_description: Newly scraped description from another source.
        min_ratio: Minimum length multiplier required to qualify as richer.
        min_delta: Minimum extra characters required to avoid tiny upgrades.

    Returns:
        bool: True if the incoming description is significantly more complete.
    """
    old_len = len(existing_description.strip())
    new_len = len(candidate_description.strip())
    if new_len <= old_len:
        return False
    if old_len < DEFAULT_SHORT_DESC_THRESHOLD and (
        new_len >= DEFAULT_SHORT_DESC_MIN_UPGRADE
    ):
        return True
    return new_len >= int(old_len * min_ratio) and (new_len - old_len) >= min_delta


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

    def _connect(self) -> aiosqlite.Connection:
        """Open an SQLite connection with busy timeout for concurrency."""
        return aiosqlite.connect(
            self.db_path,
            timeout=DEFAULT_SQLITE_TIMEOUT_SECONDS,
        )

    async def init_db(self) -> None:
        """Create tables and indices, ensuring schema migration and WAL mode."""
        # Création du dossier parent si nécessaire
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._connect() as db:
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
                    contract_type TEXT,
                    content_hash TEXT,
                    loose_hash TEXT,
                    title_loc_hash TEXT,
                    notified_at TEXT,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # Migration progressive si la table existait sans les nouvelles colonnes
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
            if "title_loc_hash" not in columns:
                await db.execute(
                    "ALTER TABLE jobs ADD COLUMN title_loc_hash TEXT;"
                )
            if "notified_at" not in columns:
                await db.execute(
                    "ALTER TABLE jobs ADD COLUMN notified_at TEXT;"
                )
            if "contract_type" not in columns:
                await db.execute(
                    "ALTER TABLE jobs ADD COLUMN contract_type TEXT;"
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
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_title_loc_hash "
                "ON jobs(title_loc_hash);"
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
        notified_iso = (
            job.notified_at.isoformat() if job.notified_at is not None else None
        )

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            try:
                # Insertion protégée par la contrainte UNIQUE sur normalized_url
                cursor = await db.execute(
                    """
                    INSERT INTO jobs (
                        platform, company, title, url, normalized_url,
                        description, raw_payload, score, contract_type,
                        content_hash, loose_hash, title_loc_hash,
                        notified_at, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        job.contract_type,
                        job.content_hash,
                        job.loose_hash,
                        job.title_loc_hash,
                        notified_iso,
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
        async with self._connect() as db:
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
        async with self._connect() as db:
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
        title_loc_hash: str | None = None,
        company: str = "",
        title: str = "",
        description: str = "",
        strict_window_days: int = DEFAULT_STRICT_WINDOW_DAYS,
        loose_window_days: int = DEFAULT_LOOSE_WINDOW_DAYS,
    ) -> tuple[JobRecord | None, str | None]:
        """Detect if an incoming job matches an existing posting across platforms.

        Applies three layers:
        1. Strict fingerprint (company + title + location) within 30 days.
        2. Loose fingerprint (company + title) within 14 days for named firms.
        3. Title + location fingerprint (B1) within 14 days when an employer is
           anonymized or when a specific title matches across portals.

        Args:
            content_hash: Strict fingerprint including location (if available).
            loose_hash: Loose fingerprint of company and title.
            title_loc_hash: Fingerprint of normalized title and location.
            company: Incoming employer name (used to detect anonymous posts).
            title: Incoming job title.
            description: Incoming description text for cross-company check.
            strict_window_days: Max age in days for a strict match (default 30).
            loose_window_days: Max age in days for a loose match (default 14).

        Returns:
            tuple[JobRecord | None, str | None]: The matched record and rule
                name, or (None, None) if no duplicate.
        """
        now = datetime.now(timezone.utc)
        incoming_anon = is_anonymous_company(company)
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row

            # 1. Vérification stricte : Entreprise + Titre + Localisation
            if content_hash and not incoming_anon:
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

            loose_cutoff = (
                now - timedelta(days=loose_window_days)
            ).isoformat()

            # 2. Vérification tolérante : Entreprise + Titre (< 14 jours)
            if not incoming_anon:
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

            # 3. Vérification B1 : Titre + Localisation (enseigne masquée ou groupe)
            if title_loc_hash:
                cursor = await db.execute(
                    """
                    SELECT * FROM jobs
                    WHERE title_loc_hash = ? AND created_at >= ?
                    ORDER BY created_at DESC LIMIT 10;
                    """,
                    (title_loc_hash, loose_cutoff),
                )
                rows = await cursor.fetchall()
                clean_tit = clean_job_title(title, company=company)
                is_specific_title = (
                    len(clean_tit.split()) >= DEFAULT_MIN_SPECIFIC_TITLE_WORDS
                    and len(clean_tit) >= DEFAULT_MIN_SPECIFIC_TITLE_CHARS
                )

                for cand_row in rows:
                    cand = self._row_to_record(cand_row)
                    if incoming_anon or is_anonymous_company(cand.company):
                        return cand, "title_loc_anonymous"

                    if not is_specific_title:
                        continue

                    min_desc_len = min(
                        len(description.strip()),
                        len(cand.description.strip()),
                    )
                    if min_desc_len < DEFAULT_STUB_DESCRIPTION_LENGTH:
                        return cand, "title_loc"

                    if self._company_mentioned_in_pair(
                        company_a=company,
                        desc_a=description,
                        company_b=cand.company,
                        desc_b=cand.description,
                    ):
                        return cand, "title_loc"

                    similarity = compute_description_similarity(
                        description, cand.description
                    )
                    if similarity >= DEFAULT_CROSS_COMPANY_SIMILARITY:
                        return cand, "title_loc"

            return None, None

    @staticmethod
    def _company_mentioned_in_pair(
        company_a: str,
        desc_a: str,
        company_b: str,
        desc_b: str,
    ) -> bool:
        """Check if either employer brand appears inside the other's description."""
        clean_a = clean_company_name(company_a)
        clean_b = clean_company_name(company_b)
        norm_desc_a = re.sub(
            r"[^\w\s]", " ", strip_accents(desc_a).lower()
        )
        norm_desc_b = re.sub(
            r"[^\w\s]", " ", strip_accents(desc_b).lower()
        )

        if clean_a and len(clean_a) >= 2:
            if re.search(rf"\b{re.escape(clean_a)}\b", norm_desc_b):
                return True
        if clean_b and len(clean_b) >= 2:
            if re.search(rf"\b{re.escape(clean_b)}\b", norm_desc_a):
                return True
        return False

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
        updated, _ = await self.merge_duplicate_offer(
            job_id=job_id,
            platform=platform,
            url=url,
            description="",
            company="",
        )
        return updated

    async def merge_duplicate_offer(
        self,
        job_id: int,
        platform: str,
        url: str,
        description: str,
        company: str = "",
    ) -> tuple[bool, bool]:
        """Merge a duplicate offer, upgrading description and requeuing if richer.

        Adds the alternate URL to raw_payload, replaces an anonymous company
        name when possible, and replaces the stored description if the new
        version is significantly more complete (re-queuing REJECTED offers
        once for fresh Gemini scoring).

        Args:
            job_id: Primary key of target JobRecord.
            platform: Secondary source identifier.
            url: Destination URL on the secondary source.
            description: Incoming offer description text.
            company: Incoming employer name.

        Returns:
            tuple[bool, bool]: (record_updated, requeued_for_evaluation).
        """
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT title, raw_payload, url, description, company, status
                FROM jobs WHERE id = ?;
                """,
                (job_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False, False

            existing_title: str = row["title"] or ""
            raw_payload_text = row["raw_payload"]
            original_url = row["url"]
            existing_desc: str = row["description"] or ""
            existing_company: str = row["company"] or ""
            existing_status: str = row["status"]
            payload = (
                json.loads(raw_payload_text)
                if raw_payload_text is not None
                else {}
            )

            now_iso = datetime.now(timezone.utc).isoformat()
            set_clauses: list[str] = []
            params: list[object] = []
            changed = False
            requeued = False

            # Ajout de l'URL alternative si inédite
            alternate_urls: list[dict[str, str]] = payload.setdefault(
                "alternate_urls", []
            )
            url_already_present = original_url == url or any(
                entry.get("url") == url for entry in alternate_urls
            )
            if url and not url_already_present:
                alternate_urls.append(
                    {
                        "platform": platform,
                        "url": url,
                        "merged_at": now_iso,
                    }
                )
                set_clauses.append("raw_payload = ?")
                params.append(json.dumps(payload))
                changed = True

            # Remplacement d'une enseigne masquée par un vrai nom d'entreprise
            if (
                company
                and is_anonymous_company(existing_company)
                and not is_anonymous_company(company)
            ):
                loc_hint = str(payload.get("location", ""))
                s_hash, l_hash, tl_hash = generate_all_fingerprints(
                    company=company,
                    title=existing_title,
                    location=loc_hint,
                )
                set_clauses.extend(
                    [
                        "company = ?",
                        "content_hash = ?",
                        "loose_hash = ?",
                        "title_loc_hash = ?",
                    ]
                )
                params.extend([company, s_hash, l_hash, tl_hash])
                changed = True

            # Mise à jour de description si nettement plus complète (ou NEW plus longue)
            clean_new_desc = description.strip()
            significantly_richer = (
                bool(clean_new_desc)
                and is_description_significantly_richer(
                    existing_description=existing_desc,
                    candidate_description=clean_new_desc,
                )
            )
            modest_upgrade_on_new = (
                existing_status == JobStatus.NEW.value
                and len(clean_new_desc) > len(existing_desc.strip())
            )
            if significantly_richer or modest_upgrade_on_new:
                set_clauses.append("description = ?")
                params.append(clean_new_desc)
                changed = True

                # Ré-évaluation uniquement si l'offre avait été rejetée sur texte pauvre
                if (
                    significantly_richer
                    and existing_status == JobStatus.REJECTED.value
                ):
                    set_clauses.extend(
                        [
                            "status = ?",
                            "score = ?",
                            "notified_at = ?",
                        ]
                    )
                    params.extend([JobStatus.NEW.value, None, None])
                    requeued = True

            if not changed:
                return False, False

            set_clauses.append("updated_at = ?")
            params.append(now_iso)
            params.append(job_id)

            update_cursor = await db.execute(
                f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?;",
                tuple(params),
            )
            await db.commit()
            return update_cursor.rowcount > 0, requeued

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
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY id ASC LIMIT ?;",
                (status.value, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    async def claim_pending_new_jobs(
        self,
        limit: int = 100,
        job_ids: list[int] | None = None,
        stale_evaluating_minutes: int = DEFAULT_STALE_EVALUATING_MINUTES,
    ) -> list[JobRecord]:
        """Atomically claim NEW jobs by transitioning them to EVALUATING.

        Prevents two concurrent orchestrator processes from evaluating and
        notifying the same job records in parallel. Stale EVALUATING rows
        older than stale_evaluating_minutes are recovered back to NEW.

        Args:
            limit: Maximum number of NEW rows to claim in one batch.
            job_ids: Optional subset of job IDs to restrict claiming to.
            stale_evaluating_minutes: Timeout before stale locks recover.

        Returns:
            list[JobRecord]: Claimed job records ready for exclusive scoring.
        """
        if job_ids is not None and not job_ids:
            return []

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        stale_cutoff = (
            now - timedelta(minutes=stale_evaluating_minutes)
        ).isoformat()

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE;")
            try:
                # Reprise automatique des verrous expirés (après crash éventuel)
                await db.execute(
                    """
                    UPDATE jobs
                    SET status = ?, updated_at = ?
                    WHERE status = ? AND updated_at < ?;
                    """,
                    (
                        JobStatus.NEW.value,
                        now_iso,
                        JobStatus.EVALUATING.value,
                        stale_cutoff,
                    ),
                )

                if job_ids is not None:
                    placeholders = ", ".join("?" for _ in job_ids)
                    select_sql = (
                        "SELECT * FROM jobs "
                        f"WHERE status = ? AND id IN ({placeholders}) "
                        "ORDER BY id ASC LIMIT ?;"
                    )
                    select_params: tuple[object, ...] = (
                        JobStatus.NEW.value,
                        *job_ids,
                        limit,
                    )
                else:
                    select_sql = (
                        "SELECT * FROM jobs "
                        "WHERE status = ? ORDER BY id ASC LIMIT ?;"
                    )
                    select_params = (JobStatus.NEW.value, limit)

                cursor = await db.execute(select_sql, select_params)
                rows = await cursor.fetchall()
                if not rows:
                    await db.commit()
                    return []

                claimed_ids = [int(row["id"]) for row in rows]
                id_placeholders = ", ".join("?" for _ in claimed_ids)
                await db.execute(
                    f"""
                    UPDATE jobs
                    SET status = ?, updated_at = ?
                    WHERE status = ? AND id IN ({id_placeholders});
                    """,
                    (
                        JobStatus.EVALUATING.value,
                        now_iso,
                        JobStatus.NEW.value,
                        *claimed_ids,
                    ),
                )
                await db.commit()

                records = [self._row_to_record(row) for row in rows]
                return [
                    rec.model_copy(
                        update={
                            "status": JobStatus.EVALUATING,
                            "updated_at": now,
                        }
                    )
                    for rec in records
                ]
            except Exception:
                await db.rollback()
                raise

    async def release_claimed_job(self, job_id: int) -> bool:
        """Reset an EVALUATING job back to NEW after an unexpected error.

        Args:
            job_id: SQLite primary key.

        Returns:
            bool: True if row transitioned from EVALUATING back to NEW.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self._connect() as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?
                WHERE id = ? AND status = ?;
                """,
                (
                    JobStatus.NEW.value,
                    now_iso,
                    job_id,
                    JobStatus.EVALUATING.value,
                ),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def mark_job_notified(self, job_id: int) -> bool:
        """Atomically reserve the right to send a Telegram alert for a job.

        Args:
            job_id: SQLite primary key.

        Returns:
            bool: True if notified_at was previously NULL and is now stamped.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self._connect() as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET notified_at = ?, updated_at = ?
                WHERE id = ? AND notified_at IS NULL;
                """,
                (now_iso, now_iso, job_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def clear_job_notified(self, job_id: int) -> bool:
        """Clear notified_at when Telegram delivery failed so alert can retry.

        Args:
            job_id: SQLite primary key.

        Returns:
            bool: True if cleared, False otherwise.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self._connect() as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET notified_at = NULL, updated_at = ?
                WHERE id = ?;
                """,
                (now_iso, job_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def update_job_evaluation(
        self,
        job_id: int,
        score: int,
        status: JobStatus,
        company: str | None = None,
        contract_type: str | None = None,
    ) -> bool:
        """Update job score, status, and optional company/contract metadata.

        Args:
            job_id: SQLite primary key.
            score: Computed match score (0-100).
            status: New status (e.g., PENDING_REVIEW or REJECTED).
            company: Optional rectified employer name.
            contract_type: Optional extracted contract type.

        Returns:
            bool: True if a row was updated, False otherwise.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        set_clauses = ["score = ?", "status = ?", "updated_at = ?"]
        params: list[object] = [score, status.value, now_iso]

        if company:
            set_clauses.append("company = ?")
            params.append(company)
        if contract_type:
            set_clauses.append("contract_type = ?")
            params.append(contract_type)

        params.append(job_id)
        query_sql = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?;"

        async with self._connect() as db:
            cursor = await db.execute(query_sql, tuple(params))
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
        async with self._connect() as db:
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
        async with self._connect() as db:
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

    async def cleanup_existing_duplicates(self) -> dict[str, int]:
        """Recompute hashes and merge existing duplicate rows in SQLite.

        Preserves the highest-priority lifecycle status, richest description,
        real employer name, and merges all alternate URLs before removing
        redundant duplicate records.

        Returns:
            dict[str, int]: Summary counts ('scanned', 'merged_deleted',
                'hashes_refreshed').
        """
        status_priority: dict[str, int] = {
            JobStatus.APPLIED.value: 5,
            JobStatus.ACCEPTED.value: 4,
            JobStatus.PENDING_REVIEW.value: 3,
            JobStatus.NEW.value: 2,
            JobStatus.EVALUATING.value: 1,
            JobStatus.REJECTED.value: 0,
        }

        await self.init_db()
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs ORDER BY id ASC;")
            rows = await cursor.fetchall()

        records = [self._row_to_record(row) for row in rows]
        if not records:
            return {"scanned": 0, "merged_deleted": 0, "hashes_refreshed": 0}

        # Groupement par graphe de correspondance (URL normalisée ou empreintes B1)
        n = len(records)
        parent = list(range(n))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        norm_urls: list[str] = []
        strict_hashes: list[str | None] = []
        loose_hashes: list[str] = []
        title_loc_hashes: list[str | None] = []
        anon_flags: list[bool] = []
        clean_titles: list[str] = []

        for rec in records:
            loc_hint = ""
            if rec.raw_payload and isinstance(rec.raw_payload, dict):
                loc_hint = str(rec.raw_payload.get("location", ""))
            s_hash, l_hash, tl_hash = generate_all_fingerprints(
                company=rec.company,
                title=rec.title,
                location=loc_hint,
            )
            norm_urls.append(normalize_job_url(rec.url))
            strict_hashes.append(s_hash)
            loose_hashes.append(l_hash)
            title_loc_hashes.append(tl_hash)
            anon_flags.append(is_anonymous_company(rec.company))
            clean_titles.append(
                clean_job_title(rec.title, company=rec.company)
            )

        for i in range(n):
            for j in range(i + 1, n):
                if norm_urls[i] == norm_urls[j]:
                    union(i, j)
                    continue
                if (
                    not anon_flags[i]
                    and not anon_flags[j]
                    and strict_hashes[i]
                    and strict_hashes[i] == strict_hashes[j]
                ):
                    union(i, j)
                    continue
                if (
                    not anon_flags[i]
                    and not anon_flags[j]
                    and loose_hashes[i] == loose_hashes[j]
                ):
                    union(i, j)
                    continue
                if (
                    title_loc_hashes[i]
                    and title_loc_hashes[i] == title_loc_hashes[j]
                ):
                    if anon_flags[i] or anon_flags[j]:
                        union(i, j)
                        continue
                    tit = clean_titles[i]
                    specific = (
                        len(tit.split()) >= DEFAULT_MIN_SPECIFIC_TITLE_WORDS
                        and len(tit) >= DEFAULT_MIN_SPECIFIC_TITLE_CHARS
                    )
                    if not specific:
                        continue
                    min_len = min(
                        len(records[i].description.strip()),
                        len(records[j].description.strip()),
                    )
                    if (
                        min_len < DEFAULT_STUB_DESCRIPTION_LENGTH
                        or self._company_mentioned_in_pair(
                            records[i].company,
                            records[i].description,
                            records[j].company,
                            records[j].description,
                        )
                        or compute_description_similarity(
                            records[i].description, records[j].description
                        )
                        >= DEFAULT_CROSS_COMPANY_SIMILARITY
                    ):
                        union(i, j)

        clusters: dict[int, list[int]] = {}
        for idx in range(n):
            clusters.setdefault(find(idx), []).append(idx)

        merged_deleted = 0
        hashes_refreshed = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        async with self._connect() as db:
            await db.execute("BEGIN IMMEDIATE;")
            try:
                for member_indices in clusters.values():
                    # Choix de l'enregistrement maître selon statut, score et texte
                    keeper_idx = max(
                        member_indices,
                        key=lambda idx: (
                            status_priority.get(records[idx].status.value, 0),
                            records[idx].score or 0,
                            0 if anon_flags[idx] else 1,
                            len(records[idx].description.strip()),
                            -(records[idx].id or 0),
                        ),
                    )
                    keeper = records[keeper_idx]
                    assert keeper.id is not None

                    best_desc = max(
                        (records[idx].description for idx in member_indices),
                        key=lambda text: len(text.strip()),
                    )
                    best_company = keeper.company
                    if anon_flags[keeper_idx]:
                        for idx in member_indices:
                            if not anon_flags[idx]:
                                best_company = records[idx].company
                                break

                    payload = dict(keeper.raw_payload or {})
                    alt_urls: list[dict[str, str]] = list(
                        payload.get("alternate_urls", [])
                    )
                    seen_alt_urls = {keeper.url} | {
                        str(entry.get("url", "")) for entry in alt_urls
                    }
                    for idx in member_indices:
                        if idx == keeper_idx:
                            continue
                        dup = records[idx]
                        if dup.url not in seen_alt_urls:
                            alt_urls.append(
                                {
                                    "platform": dup.platform,
                                    "url": dup.url,
                                    "merged_at": now_iso,
                                }
                            )
                            seen_alt_urls.add(dup.url)
                        dup_payload = dup.raw_payload or {}
                        for extra in dup_payload.get("alternate_urls", []):
                            extra_url = str(extra.get("url", ""))
                            if extra_url and extra_url not in seen_alt_urls:
                                alt_urls.append(extra)
                                seen_alt_urls.add(extra_url)

                    if alt_urls:
                        payload["alternate_urls"] = alt_urls

                    # Suppression d'abord des doublons pour libérer normalized_url
                    for idx in member_indices:
                        if idx == keeper_idx:
                            continue
                        dup_id = records[idx].id
                        if dup_id is not None:
                            await db.execute(
                                "DELETE FROM jobs WHERE id = ?;",
                                (dup_id,),
                            )
                            merged_deleted += 1

                    loc_hint = str(payload.get("location", ""))
                    s_hash, l_hash, tl_hash = generate_all_fingerprints(
                        company=best_company,
                        title=keeper.title,
                        location=loc_hint,
                    )
                    keeper_norm_url = normalize_job_url(keeper.url)
                    notified_val = (
                        keeper.notified_at.isoformat()
                        if keeper.notified_at is not None
                        else (
                            now_iso
                            if keeper.status == JobStatus.PENDING_REVIEW
                            else None
                        )
                    )
                    await db.execute(
                        """
                        UPDATE jobs
                        SET company = ?, description = ?, raw_payload = ?,
                            normalized_url = ?, content_hash = ?,
                            loose_hash = ?, title_loc_hash = ?,
                            notified_at = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (
                            best_company,
                            best_desc,
                            json.dumps(payload) if payload else None,
                            keeper_norm_url,
                            s_hash,
                            l_hash,
                            tl_hash,
                            notified_val,
                            now_iso,
                            keeper.id,
                        ),
                    )
                    hashes_refreshed += 1

                await db.commit()
            except Exception:
                await db.rollback()
                raise

        return {
            "scanned": len(records),
            "merged_deleted": merged_deleted,
            "hashes_refreshed": hashes_refreshed,
        }

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

        row_keys = row.keys()
        contract_type = (
            row["contract_type"] if "contract_type" in row_keys else None
        )
        content_hash = (
            row["content_hash"] if "content_hash" in row_keys else None
        )
        loose_hash = (
            row["loose_hash"] if "loose_hash" in row_keys else None
        )
        title_loc_hash = (
            row["title_loc_hash"] if "title_loc_hash" in row_keys else None
        )
        notified_at_raw = (
            row["notified_at"] if "notified_at" in row_keys else None
        )
        notified_at = (
            datetime.fromisoformat(notified_at_raw)
            if notified_at_raw
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
            contract_type=contract_type,
            content_hash=content_hash,
            loose_hash=loose_hash,
            title_loc_hash=title_loc_hash,
            notified_at=notified_at,
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
