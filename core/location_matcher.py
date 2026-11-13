"""Geographic location matcher with 25km radius check and Gemini fallback."""

import json
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

from google import genai
from google.genai import types
import httpx
from pydantic import BaseModel, Field

from config.criteria import parse_search_criteria_file
from config.settings import get_settings

# API officielle Base Adresse Nationale (géocodage sans clé API)
BAN_SEARCH_URL: str = "https://api-adresse.data.gouv.fr/search/"
# Rayon maximal autorisé dans le département cible depuis le domicile (réglable)
DEFAULT_MAX_DISTANCE_KM: float = 30.0
# Coordonnées par défaut du domicile (Allauch 13190) (réglable)
DEFAULT_RESIDENCE_LABEL: str = "Allauch (13190), France"
DEFAULT_RESIDENCE_COORDS: tuple[float, float] = (43.341016, 5.485377)
DEFAULT_TARGET_DEPARTMENT: str = "13"

# Mots-clés géographiques généraux (ne valent pas validation directe de ville)
GENERIC_LOCATION_TOKENS: frozenset[str] = frozenset(
    {
        "13",
        "bouches du rhone",
        "bouches-du-rhone",
        "paca",
        "provence alpes cote d azur",
        "remote",
        "teletravail",
    }
)
REMOTE_TOKENS: tuple[str, ...] = (
    "remote",
    "teletravail",
    "full remote",
    "distanciel",
)


class GeminiLocationCheck(BaseModel):
    """Structured geography verification returned by Gemini Flash.

    Attributes:
        in_target_department: True if located in target department (e.g. 13).
        distance_km: Estimated straight-line distance in km from residence.
        normalized_city: Canonical city name detected.
        reason: Brief geographic explanation.
    """

    in_target_department: bool = Field(
        default=False,
        description="True if the location belongs to department 13.",
    )
    distance_km: float | None = Field(
        default=None,
        description="Estimated distance in km from candidate residence.",
    )
    normalized_city: str = Field(
        default="",
        description="Canonical municipality name extracted.",
    )
    reason: str = Field(
        default="",
        description="Short justification of department and distance.",
    )


def normalize_geo_text(text: str) -> str:
    """Normalize text by stripping accents, punctuation, and extra spaces.

    Args:
        text: Raw location or city text.

    Returns:
        str: Lowercased ASCII-folded location string.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_folded = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).lower()
    cleaned = re.sub(r"[-_/,().]+", " ", ascii_folded)
    return re.sub(r"\s+", " ", cleaned).strip()


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compute great-circle distance between two WGS84 coordinates in km.

    Args:
        lat1: Origin latitude in degrees.
        lon1: Origin longitude in degrees.
        lat2: Destination latitude in degrees.
        lon2: Destination longitude in degrees.

    Returns:
        float: Distance in kilometers.
    """
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c


class LocationMatcher:
    """Shared location validator checking criteria list and 25km dept radius.

    Validates job locations against:
    1. Explicit cities or remote tags in search_criteria.txt.
    2. For other localities inside the target department (13), verifies
       that distance to candidate residence (profile.md) is <= max_distance_km,
       using French BAN geocoding + Gemini Flash fallback + JSON disk cache.
    """

    def __init__(
        self,
        criteria_locations: list[str] | None = None,
        residence_label: str | None = None,
        residence_coords: tuple[float, float] = DEFAULT_RESIDENCE_COORDS,
        max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
        target_department: str = DEFAULT_TARGET_DEPARTMENT,
        cache_path: Path | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the shared location matcher.

        Args:
            criteria_locations: Optional explicit target locations list.
            residence_label: Candidate home city/postcode for radius check.
            residence_coords: Latitude/longitude pair of home location.
            max_distance_km: Radius threshold in km within target department.
            target_department: French department code (default '13').
            cache_path: JSON cache file storing resolved location decisions.
            api_key: Optional Gemini API key for ambiguous labels.
            model_name: Gemini model identifier for fallback checks.
        """
        settings = get_settings()
        if criteria_locations is None:
            criteria = parse_search_criteria_file(settings.search_criteria_path)
            criteria_locations = criteria.locations

        self.criteria_locations: list[str] = list(criteria_locations)
        self.max_distance_km: float = max_distance_km
        self.target_department: str = target_department.strip()

        self.residence_label: str = (
            residence_label
            or self._load_residence_from_profile(
                settings.candidate_profile_path
            )
            or DEFAULT_RESIDENCE_LABEL
        )
        self.residence_coords: tuple[float, float] = residence_coords
        self._residence_resolved: bool = (
            residence_label is None
            and "allauch" in normalize_geo_text(self.residence_label)
        )

        self.cache_path: Path = (
            cache_path
            if cache_path is not None
            else settings.database_path.parent / "location_cache.json"
        )
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

        resolved_key = (
            api_key
            if api_key is not None
            else settings.gemini_api_key.get_secret_value()
        )
        self._api_key: str = resolved_key
        self.model_name: str = model_name or settings.gemini_model
        self._genai_client: genai.Client | None = (
            genai.Client(api_key=self._api_key)
            if self._api_key and self._api_key != "DUMMY_KEY"
            else None
        )

    @staticmethod
    def _load_residence_from_profile(profile_path: Path) -> str | None:
        """Extract candidate residence city from profile.md if present.

        Args:
            profile_path: Path to candidate Markdown profile.

        Returns:
            str | None: Extracted location line or None.
        """
        if not profile_path.exists():
            return None
        try:
            content = profile_path.read_text(encoding="utf-8")
        except OSError:
            return None
        match = re.search(
            r"Current Location:\*?\*?\s*([^\n]+)",
            content,
            re.IGNORECASE,
        )
        if not match:
            return None
        raw = match.group(1).strip()
        # Suppression des mentions parenthésées secondaires en fin de ligne
        cleaned = re.split(r"\s*\((?=[A-Za-z])", raw, maxsplit=1)[0].strip()
        return cleaned or raw

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Load persisted location validation decisions from disk."""
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        """Persist location decisions atomically to JSON cache file."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _cache_key(self, raw_location: str) -> str:
        """Build deterministic cache key scoped to residence and radius."""
        norm_loc = normalize_geo_text(raw_location)
        norm_res = normalize_geo_text(self.residence_label)
        return (
            f"{norm_loc}|{self.target_department}|"
            f"{norm_res}|{self.max_distance_km:.1f}"
        )

    def _split_targets(
        self, extra_locations: list[str] | None = None
    ) -> tuple[list[str], bool]:
        """Separate explicit city names from generic department/remote flags.

        Args:
            extra_locations: Optional extra allowed tokens from a scraper.

        Returns:
            tuple[list[str], bool]: Explicit normalized city strings and
                whether remote jobs are accepted.
        """
        combined = list(self.criteria_locations)
        if extra_locations:
            combined.extend(extra_locations)

        explicit_cities: list[str] = []
        allow_remote = False
        for token in combined:
            norm = normalize_geo_text(token)
            if not norm:
                continue
            if any(rt in norm for rt in REMOTE_TOKENS):
                allow_remote = True
                continue
            if norm in GENERIC_LOCATION_TOKENS:
                continue
            if norm not in explicit_cities:
                explicit_cities.append(norm)
        return explicit_cities, allow_remote

    @staticmethod
    def _contains_token(haystack_norm: str, token_norm: str) -> bool:
        """Check whether normalized token appears on word boundaries."""
        if not haystack_norm or not token_norm:
            return False
        pattern = rf"(?:^|\s){re.escape(token_norm)}(?:\s|$)"
        return bool(re.search(pattern, haystack_norm))

    def _matches_explicit_criteria(
        self,
        raw_location: str,
        extra_locations: list[str] | None = None,
    ) -> bool | None:
        """Return True on direct match, None if 13/radius verification needed.

        Args:
            raw_location: Raw job location string.
            extra_locations: Optional extra allowed location tokens.

        Returns:
            bool | None: True if explicitly allowed, None if further check needed.
        """
        norm_loc = normalize_geo_text(raw_location)
        if not norm_loc:
            return False

        explicit_cities, allow_remote = self._split_targets(extra_locations)
        if allow_remote and any(rt in norm_loc for rt in REMOTE_TOKENS):
            return True

        for city in explicit_cities:
            if self._contains_token(norm_loc, city):
                return True

        return None

    async def _ensure_residence_coords(
        self, client: httpx.AsyncClient
    ) -> tuple[float, float]:
        """Resolve candidate home coordinates once via BAN if needed."""
        if self._residence_resolved:
            return self.residence_coords

        try:
            resp = await client.get(
                BAN_SEARCH_URL,
                params={"q": self.residence_label, "limit": 1},
            )
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                if features:
                    coords = (
                        features[0].get("geometry", {}).get("coordinates", [])
                    )
                    if len(coords) == 2:
                        lon, lat = float(coords[0]), float(coords[1])
                        self.residence_coords = (lat, lon)
        except Exception:
            pass
        finally:
            self._residence_resolved = True
        return self.residence_coords

    @staticmethod
    def _clean_municipality_query(raw_location: str) -> str:
        """Extract clean city/postcode query string for French BAN lookup."""
        cleaned = re.sub(
            r"\b(france|paca|bouches[- ]du[- ]rh[oô]ne|area|cedex)\b"
            r"|provence[- ]alpes[- ]c[oô]te[- ]d['’ ]?azur",
            " ",
            raw_location,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[(),/]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    async def _check_via_ban(
        self,
        client: httpx.AsyncClient,
        raw_location: str,
    ) -> dict[str, Any] | None:
        """Geocode a French location via BAN and compute distance in km.

        Args:
            client: Active async HTTP client.
            raw_location: Job location string to verify.

        Returns:
            dict[str, Any] | None: Decision payload or None if unresolved.
        """
        home_lat, home_lon = await self._ensure_residence_coords(client)
        query = self._clean_municipality_query(raw_location) or raw_location

        search_variants: tuple[dict[str, str | int], ...] = (
            {"q": query, "type": "municipality", "limit": 5},
            {"q": query, "limit": 5},
        )
        for query_params in search_variants:
            try:
                resp = await client.get(BAN_SEARCH_URL, params=query_params)
                if resp.status_code != 200:
                    continue
                features = resp.json().get("features", [])
                if not features:
                    continue

                # Priorité à une correspondance dans le département cible (13)
                target_feature = None
                for feat in features:
                    props = feat.get("properties", {})
                    ctx = str(props.get("context", ""))
                    postcode = str(props.get("postcode", ""))
                    if ctx.startswith(f"{self.target_department},") or (
                        postcode.startswith(self.target_department)
                    ):
                        target_feature = feat
                        break

                if target_feature is not None:
                    coords = (
                        target_feature.get("geometry", {}).get("coordinates", [])
                    )
                    if len(coords) == 2:
                        lon, lat = float(coords[0]), float(coords[1])
                        dist_km = round(
                            haversine_km(home_lat, home_lon, lat, lon), 2
                        )
                        city_name = str(
                            target_feature.get("properties", {}).get(
                                "city", query
                            )
                        )
                        return {
                            "allowed": dist_km <= self.max_distance_km,
                            "in_department": True,
                            "distance_km": dist_km,
                            "city": city_name,
                            "source": "ban",
                        }

                # Si le premier résultat a un score élevé hors 13, on rejette
                top_props = features[0].get("properties", {})
                top_score = float(top_props.get("score", 0.0))
                if top_score >= 0.55:
                    return {
                        "allowed": False,
                        "in_department": False,
                        "distance_km": None,
                        "city": str(top_props.get("city", query)),
                        "source": "ban",
                    }
            except Exception:
                continue

        return None

    async def _check_via_gemini(
        self, raw_location: str
    ) -> dict[str, Any] | None:
        """Verify department 13 membership and <=25km radius via Gemini Flash.

        Args:
            raw_location: Ambiguous or composite location string.

        Returns:
            dict[str, Any] | None: Decision payload or None if unavailable.
        """
        if self._genai_client is None:
            return None

        prompt = (
            f"Evaluate the job location: '{raw_location}'.\n"
            f"1. Is it located in French department '{self.target_department}' "
            f"(Bouches-du-Rhône)?\n"
            f"2. What is its approximate distance in kilometers (straight-line) "
            f"from the candidate residence '{self.residence_label}'?\n"
            "Return strict JSON matching the schema."
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiLocationCheck,
            temperature=0.0,
        )

        models = [self.model_name]
        for fb in ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite"):
            if fb not in models:
                models.append(fb)

        for model_id in models:
            try:
                res = await self._genai_client.aio.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config,
                )
                parsed: GeminiLocationCheck | None = None
                if hasattr(res, "parsed") and isinstance(
                    res.parsed, GeminiLocationCheck
                ):
                    parsed = res.parsed
                elif res.text:
                    parsed = GeminiLocationCheck.model_validate_json(res.text)

                if parsed is not None:
                    dist = parsed.distance_km
                    allowed = bool(
                        parsed.in_target_department
                        and dist is not None
                        and dist <= self.max_distance_km
                    )
                    return {
                        "allowed": allowed,
                        "in_department": parsed.in_target_department,
                        "distance_km": dist,
                        "city": parsed.normalized_city or raw_location,
                        "source": "gemini",
                        "reason": parsed.reason,
                    }
            except Exception:
                continue

        return None

    async def is_location_allowed(
        self,
        raw_location: str,
        extra_locations: list[str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> bool:
        """Check if a job location matches criteria or <=25km in department 13.

        Args:
            raw_location: Raw location string extracted from an offer.
            extra_locations: Optional extra allowed location tokens.
            client: Optional reusable async HTTP client.

        Returns:
            bool: True if explicitly listed, remote, or in dept 13 within radius.
        """
        direct = self._matches_explicit_criteria(
            raw_location=raw_location,
            extra_locations=extra_locations,
        )
        if direct is not None:
            return direct

        # Si le département 13 / Bouches-du-Rhône n'est pas configuré dans les
        # critères, seules les villes explicitement listées sont acceptées.
        norm_criteria = {
            normalize_geo_text(loc) for loc in self.criteria_locations
        }
        dept_enabled = any(
            tok in norm_criteria
            for tok in ("13", "bouches du rhone", "bouches-du-rhone")
        )
        if not dept_enabled and self.criteria_locations:
            return False

        # Si la chaîne désigne uniquement le département 13 sans ville précise
        norm_raw = normalize_geo_text(raw_location)
        if norm_raw in ("13", "bouches du rhone", "bouches du rhone 13"):
            return True

        cache_key = self._cache_key(raw_location)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return bool(cached.get("allowed", False))

        decision: dict[str, Any] | None = None
        if client is not None:
            decision = await self._check_via_ban(client, raw_location)
        else:
            async with httpx.AsyncClient(timeout=8.0) as local_client:
                decision = await self._check_via_ban(local_client, raw_location)

        if decision is None:
            decision = await self._check_via_gemini(raw_location)

        if decision is None:
            # Repli ultime : si '(13)' / 'Bouches-du-Rhône' explicite sans ville
            norm_loc = normalize_geo_text(raw_location)
            explicit_13 = bool(
                re.search(r"\b13\d{3}\b|\b13\b", norm_loc)
                or "bouches du rhone" in norm_loc
            )
            decision = {
                "allowed": explicit_13,
                "in_department": explicit_13,
                "distance_km": None,
                "source": "fallback",
            }

        self._cache[cache_key] = decision
        self._save_cache()
        return bool(decision.get("allowed", False))
