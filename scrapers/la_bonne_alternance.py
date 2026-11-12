"""Scraper implementation for La bonne alternance official API."""

from collections.abc import AsyncIterator
import html
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from config.settings import get_settings
from core.location_matcher import (
    LocationMatcher,
    haversine_km,
    normalize_geo_text,
)
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

LBA_SEARCH_URL: str = (
    "https://api.apprentissage.beta.gouv.fr/api/job/v1/search"
)

# Codes ROME IT/IA/Cybersécurité/Systèmes ciblés pour recruteurs_lba (réglable)
DEFAULT_SPONTANEOUS_IT_ROMES: str = (
    "M1801,M1802,M1803,M1804,M1805,M1806,M1810,M1818,M1855,M1861,M1881,I1401"
)
# Préfixes NAF priorisant les entreprises du secteur numérique / ingénierie
IT_NAF_PREFIXES: tuple[str, ...] = ("62.", "63.", "58.2", "71.12", "72.1")

# Préfixes ROME 4.0 couvrant l'informatique, l'IA, la data et la cybersécurité
DEFAULT_IT_ROME_PREFIXES: tuple[str, ...] = ("M18", "I14")

# Termes techniques complémentaires pour filtrer les offres hors préfixe M18/I14
TECH_SIGNAL_TOKENS: tuple[str, ...] = (
    "python",
    "devops",
    "devsecops",
    "cybersecurite",
    "cyber",
    "soc",
    "csirt",
    "pentest",
    "ia",
    "intelligence artificielle",
    "machine learning",
    "data",
    "backend",
    "software",
    "logiciel",
    "developpeur",
    "cloud",
    "kubernetes",
    "docker",
    "linux",
    "securite informatique",
    "embarque",
)


def _clean_html(raw_text: str | None) -> str:
    """Decode HTML entities, strip tags, and normalize whitespace."""
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


class SpontaneousRecruiter(BaseModel):
    """Company identified by La bonne alternance for spontaneous applications.

    Attributes:
        company: Workplace company or brand name.
        naf_code: French NAF industry code if reported.
        naf_label: Human-readable sector label.
        address: Street/city location string.
        size: Headcount range string.
        distance_km: Straight-line distance in km from candidate residence.
        apply_url: Direct La bonne alternance application page URL.
        website: Optional corporate website URL.
        is_it_sector: True when NAF sector matches tech/IT/engineering.
    """

    company: str = Field(..., description="Company name.")
    naf_code: str = Field(default="", description="NAF code.")
    naf_label: str = Field(default="", description="Sector description.")
    address: str = Field(default="", description="Office address.")
    size: str = Field(default="", description="Headcount range.")
    distance_km: float | None = Field(
        default=None,
        description="Distance from candidate residence in km.",
    )
    apply_url: str = Field(..., description="Direct application URL.")
    website: str | None = Field(default=None, description="Company website.")
    is_it_sector: bool = Field(default=False, description="IT NAF sector flag.")


class LaBonneAlternanceScraper(BaseScraper):
    """Scraper querying La bonne alternance API for real apprenticeship offers.

    Excludes unposted spontaneous recruiter suggestions (recruteurs_lba) and
    filters active offers within the configured geographic radius around the
    candidate residence without enforcing a diploma level restriction.

    Attributes:
        radius: Search radius in kilometers around residence (tunable).
        timeout_seconds: HTTP request timeout in seconds.
        it_rome_prefixes: ROME code prefixes identifying IT/tech roles.
        location_matcher: Shared geo matcher enforcing 25km dept 13 rules.
    """

    def __init__(
        self,
        api_key: str | None = None,
        radius: int = 25,
        timeout_seconds: float = 20.0,
        it_rome_prefixes: tuple[str, ...] = DEFAULT_IT_ROME_PREFIXES,
        location_matcher: LocationMatcher | None = None,
    ) -> None:
        """Initialize the La bonne alternance scraper.

        Args:
            api_key: Optional Bearer token from api.apprentissage.beta.gouv.fr.
            radius: Geographic search radius in km (0-200).
            timeout_seconds: Network timeout for API calls.
            it_rome_prefixes: ROME code prefixes retained automatically.
            location_matcher: Shared geographic matcher instance.
        """
        super().__init__(name="la_bonne_alternance")
        settings = get_settings()
        self._api_key: str = (
            api_key
            if api_key is not None
            else settings.lba_api_key.get_secret_value()
        ).strip()
        self.radius = max(0, min(radius, 200))
        self.timeout_seconds = timeout_seconds
        self.it_rome_prefixes = it_rome_prefixes
        self.location_matcher = location_matcher or LocationMatcher()
        self._recruiters_cache: list[SpontaneousRecruiter] = []
        self._recruiters_cached_at: float = 0.0

    @staticmethod
    def _matches_word_token(haystack_norm: str, token_norm: str) -> bool:
        """Return True when normalized token matches on word boundaries."""
        if not haystack_norm or not token_norm:
            return False
        pattern = rf"(?:^|\s){re.escape(token_norm)}(?:\s|$)"
        return bool(re.search(pattern, haystack_norm))

    def _is_relevant_it_offer(
        self,
        title: str,
        description: str,
        rome_codes: list[str],
        keywords: list[str],
    ) -> bool:
        """Keep offers matching IT ROME codes or candidate technical keywords.

        Args:
            title: Job posting title.
            description: Cleaned job description text.
            rome_codes: France Travail ROME v4 codes attached to the offer.
            keywords: Active search keywords from search_criteria.txt.

        Returns:
            bool: True if offer belongs to IT, AI, dev, or cybersecurity.
        """
        if any(
            str(code).upper().startswith(self.it_rome_prefixes)
            for code in rome_codes
        ):
            return True

        norm_title = normalize_geo_text(title)

        # Mot-clé explicite dans le titre (hors terme générique 'systèmes')
        for kw in keywords:
            norm_kw = normalize_geo_text(kw)
            if not norm_kw or norm_kw in ("systemes", "systeme"):
                continue
            if self._matches_word_token(norm_title, norm_kw):
                return True

        return any(
            self._matches_word_token(norm_title, tok)
            for tok in TECH_SIGNAL_TOKENS
        )

    @staticmethod
    def _format_description(item: dict[str, Any]) -> str:
        """Assemble full structured offer description for Gemini scoring."""
        offer = item.get("offer") or {}
        workplace = item.get("workplace") or {}
        contract = item.get("contract") or {}
        loc_info = workplace.get("location") or {}

        title = _clean_html(str(offer.get("title") or "Alternance"))
        company = _clean_html(
            str(
                workplace.get("name")
                or workplace.get("brand")
                or workplace.get("legal_name")
                or "Entreprise confidentielle"
            )
        )
        address = _clean_html(str(loc_info.get("address") or ""))
        contract_types = contract.get("type") or ["Alternance"]
        contract_str = ", ".join(str(t) for t in contract_types)
        duration = contract.get("duration")
        remote = contract.get("remote")
        rome_codes = offer.get("rome_codes") or []

        desired_skills = [
            _clean_html(str(s))
            for s in (offer.get("desired_skills") or [])
            if s
        ]
        target_skills = [
            _clean_html(str(s))
            for s in (offer.get("to_be_acquired_skills") or [])
            if s
        ]

        offer_desc = _clean_html(str(offer.get("description") or ""))
        wp_desc = _clean_html(str(workplace.get("description") or ""))

        loc_label = address or "Bouches-du-Rhône"
        lines = [
            f"{title} chez {company}",
            f"Lieu : {loc_label} | Contrat : Alternance ({contract_str})",
        ]
        if duration:
            lines.append(f"Durée : {duration} mois")
        if remote:
            lines.append(f"Télétravail : {remote}")
        if rome_codes:
            lines.append(f"Codes ROME : {', '.join(str(c) for c in rome_codes)}")
        if desired_skills:
            lines.append(f"Compétences attendues : {', '.join(desired_skills[:8])}")
        if target_skills:
            lines.append(f"Compétences visées : {', '.join(target_skills[:8])}")
        lines.append(f"\nDescription du poste :\n{offer_desc or title}")
        if wp_desc:
            lines.append(f"\nÀ propos de l'entreprise :\n{wp_desc}")
        return "\n".join(lines)

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch real alternance offers from La bonne alternance API.

        Args:
            keywords: Target job keywords for IT relevance filtering.
            locations: Geographic filters validated by LocationMatcher.
            limit: Maximum number of offers to yield.

        Yields:
            RawJobListing: Normalized apprenticeship offer record.
        """
        if not self._api_key:
            print(
                "ℹ️ [LBA] Clé LBA_API_KEY absente dans .env. "
                "Scraper La bonne alternance ignoré."
            )
            return

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(
            headers=headers,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            lat, lon = await self.location_matcher._ensure_residence_coords(
                client
            )
            params: dict[str, str | float | int] = {
                "latitude": lat,
                "longitude": lon,
                "radius": self.radius,
            }

            try:
                response = await client.get(LBA_SEARCH_URL, params=params)
                if response.status_code in (401, 403):
                    print(
                        "⚠️ [LBA] Clé LBA_API_KEY refusée "
                        f"(HTTP {response.status_code})."
                    )
                    return
                if response.status_code != 200:
                    print(
                        f"⚠️ [LBA] Réponse API inattendue : "
                        f"HTTP {response.status_code}"
                    )
                    return
                payload = response.json()
            except (httpx.HTTPError, ValueError) as err:
                print(f"⚠️ [LBA] Erreur réseau API : {err}")
                return

            raw_jobs = payload.get("jobs", [])
            if not isinstance(raw_jobs, list):
                return

            yielded_count = 0
            seen_ids: set[str] = set()

            for item in raw_jobs:
                if yielded_count >= limit:
                    break
                if not isinstance(item, dict):
                    continue

                ident = item.get("identifier") or {}
                partner_label = str(ident.get("partner_label") or "").strip()
                # Choix 1 : exclure les simples suggestions de candidatures spontanées
                if partner_label.lower() == "recruteurs_lba":
                    continue

                offer_id = str(ident.get("id") or "").strip()
                offer = item.get("offer") or {}
                title = _clean_html(str(offer.get("title") or ""))
                if not offer_id or not title or offer_id in seen_ids:
                    continue
                if is_excluded_title(title):
                    continue

                raw_desc = _clean_html(str(offer.get("description") or ""))
                rome_codes = [
                    str(c) for c in (offer.get("rome_codes") or []) if c
                ]
                if not self._is_relevant_it_offer(
                    title=title,
                    description=raw_desc,
                    rome_codes=rome_codes,
                    keywords=keywords,
                ):
                    continue

                workplace = item.get("workplace") or {}
                company = _clean_html(
                    str(
                        workplace.get("name")
                        or workplace.get("brand")
                        or workplace.get("legal_name")
                        or "Entreprise confidentielle"
                    )
                )
                loc_info = workplace.get("location") or {}
                address = _clean_html(str(loc_info.get("address") or ""))
                contract = item.get("contract") or {}
                remote_val = str(contract.get("remote") or "").lower()

                if address and remote_val != "full":
                    allowed = await self.location_matcher.is_location_allowed(
                        raw_location=address,
                        extra_locations=locations,
                        client=client,
                    )
                    if not allowed:
                        continue

                apply_info = item.get("apply") or {}
                apply_url = str(apply_info.get("url") or "").strip()
                if not apply_url:
                    apply_url = (
                        "https://labonnealternance.apprentissage.beta.gouv.fr"
                        f"/emploi/{partner_label or 'offres_emploi_lba'}/{offer_id}"
                    )

                seen_ids.add(offer_id)
                full_description = self._format_description(item)

                yield RawJobListing(
                    platform=self.name,
                    company=company,
                    title=title,
                    url=apply_url,
                    description=full_description,
                    location=address or "Bouches-du-Rhône",
                    raw_payload={
                        "offer_id": offer_id,
                        "partner_label": partner_label,
                        "partner_job_id": ident.get("partner_job_id"),
                        "rome_codes": rome_codes,
                        "contract_type": "Alternance",
                        "is_easy_apply": False,
                        "apply_url": apply_url,
                    },
                )
                yielded_count += 1

    async def fetch_spontaneous_recruiters(
        self,
        it_only: bool = True,
        force_refresh: bool = False,
        cache_ttl_seconds: float = 900.0,
    ) -> list[SpontaneousRecruiter]:
        """Fetch nearby recruiters_lba companies within 25 km for /spontanees.

        Args:
            it_only: Keep only companies whose NAF code is in IT/engineering.
            force_refresh: Bypass in-memory cache when True.
            cache_ttl_seconds: Cache validity duration in seconds.

        Returns:
            list[SpontaneousRecruiter]: Sorted list of candidate target firms.
        """
        if not self._api_key:
            return []

        now = time.monotonic()
        if (
            not force_refresh
            and self._recruiters_cache
            and (now - self._recruiters_cached_at) < cache_ttl_seconds
        ):
            items = self._recruiters_cache
            return [r for r in items if not it_only or r.is_it_sector]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(
            headers=headers,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            lat, lon = await self.location_matcher._ensure_residence_coords(
                client
            )
            params: dict[str, str | float | int] = {
                "romes": DEFAULT_SPONTANEOUS_IT_ROMES,
                "latitude": lat,
                "longitude": lon,
                "radius": self.radius,
            }
            try:
                response = await client.get(LBA_SEARCH_URL, params=params)
                if response.status_code != 200:
                    return []
                payload = response.json()
            except (httpx.HTTPError, ValueError):
                return []

        raw_recs = payload.get("recruiters", [])
        if not isinstance(raw_recs, list):
            return []

        parsed: list[SpontaneousRecruiter] = []
        seen_keys: set[str] = set()

        for rec in raw_recs:
            if not isinstance(rec, dict):
                continue
            wp = rec.get("workplace") or {}
            company = _clean_html(
                str(
                    wp.get("name")
                    or wp.get("brand")
                    or wp.get("legal_name")
                    or ""
                )
            )
            if not company:
                continue

            siret = str(wp.get("siret") or "").strip()
            dedup_key = siret or normalize_geo_text(company)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            domain = wp.get("domain") or {}
            naf = domain.get("naf") or {}
            naf_code = str(naf.get("code") or "").strip()
            naf_label = _clean_html(str(naf.get("label") or ""))
            is_it = naf_code.startswith(IT_NAF_PREFIXES) or any(
                tok in normalize_geo_text(naf_label)
                for tok in (
                    "informatique",
                    "logiciel",
                    "programmation",
                    "portails internet",
                    "traitement de donnees",
                )
            )

            loc_info = wp.get("location") or {}
            address = _clean_html(str(loc_info.get("address") or ""))
            coords = (loc_info.get("geopoint") or {}).get("coordinates") or []
            dist_km: float | None = None
            if len(coords) == 2:
                rec_lon, rec_lat = float(coords[0]), float(coords[1])
                dist_km = round(haversine_km(lat, lon, rec_lat, rec_lon), 1)
                if dist_km > float(self.radius):
                    continue

            apply_url = str((rec.get("apply") or {}).get("url") or "").strip()
            if not apply_url:
                continue
            website = wp.get("website")
            website_str = str(website).strip() if website else None

            parsed.append(
                SpontaneousRecruiter(
                    company=company,
                    naf_code=naf_code,
                    naf_label=naf_label,
                    address=address,
                    size=str(wp.get("size") or "").strip(),
                    distance_km=dist_km,
                    apply_url=apply_url,
                    website=website_str,
                    is_it_sector=is_it,
                )
            )

        # Tri : entreprises IT en priorité, puis par distance croissante
        parsed.sort(
            key=lambda r: (
                not r.is_it_sector,
                r.distance_km if r.distance_km is not None else 999.0,
                r.company.lower(),
            )
        )
        self._recruiters_cache = parsed
        self._recruiters_cached_at = now

        if it_only:
            it_filtered = [r for r in parsed if r.is_it_sector]
            return it_filtered or parsed
        return parsed
