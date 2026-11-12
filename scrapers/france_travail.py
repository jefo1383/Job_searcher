"""Scraper for France Travail supporting official API v2 and web fallback."""

import asyncio
from collections.abc import AsyncIterator
import html
import re
import time
from typing import Any

import httpx

from config.settings import get_settings
from core.location_matcher import LocationMatcher, normalize_geo_text
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# Endpoints officiels de l'API France Travail
FRANCE_TRAVAIL_TOKEN_URL: str = (
    "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
    "?realm=%2Fpartenaire"
)
FRANCE_TRAVAIL_SEARCH_URL: str = (
    "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
)
DEFAULT_OAUTH_SCOPE: str = "api_offresdemploiv2 o2dsoffre"

# Endpoints publics du portail candidat (repli sans clé API)
FT_PUBLIC_SEARCH_URL: str = (
    "https://candidat.francetravail.fr/offres/recherche"
)
FT_PUBLIC_DETAIL_API: str = (
    "https://candidat.francetravail.fr/api-descriptifoffre/offre"
)
FT_PUBLIC_DETAIL_PAGE: str = (
    "https://candidat.francetravail.fr/offres/recherche/detail"
)
DEFAULT_PUBLIC_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


def _clean_html(raw_text: str | None) -> str:
    """Strip HTML tags, decode entities, and collapse whitespace."""
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


def _format_public_offer_description(
    detail: dict[str, Any],
    fallback_title: str,
    fallback_company: str,
    fallback_location: str,
    fallback_contract: str,
    fallback_snippet: str,
) -> str:
    """Build full description text from France Travail public detail JSON.

    Args:
        detail: JSON payload from api-descriptifoffre/offre/{id}.
        fallback_title: Title parsed from the search card.
        fallback_company: Company name parsed from the search card.
        fallback_location: Location label parsed from the search card.
        fallback_contract: Contract summary parsed from the search card.
        fallback_snippet: Short excerpt from the search card.

    Returns:
        str: Rich plain-text offer summary and full body for AI scoring.
    """
    title = str(detail.get("intituleOffre") or fallback_title).strip()
    etab = detail.get("etablissement") or {}
    company = str(
        etab.get("libelleEnseigne") or fallback_company
    ).strip()
    lieu_info = detail.get("lieuTravail") or {}
    location = str(
        lieu_info.get("syntheseLieuTravail")
        or detail.get("libelleCodeLieuTravail")
        or fallback_location
    ).strip()
    contract = str(
        detail.get("syntheseContrat") or fallback_contract
    ).strip()
    salary = str(detail.get("salaire") or "").strip()

    exp_info = detail.get("experienceProfil") or {}
    experience = str(exp_info.get("experience") or "").strip()

    raw_skills = detail.get("listeCompetence") or []
    skills = [
        str(item.get("libelleCompetence")).strip()
        for item in raw_skills
        if isinstance(item, dict) and item.get("libelleCompetence")
    ]

    body = str(detail.get("descriptif") or fallback_snippet).strip()

    lines = [
        f"{title} chez {company}",
        f"Lieu : {location} | Contrat : {contract}",
    ]
    if salary:
        lines.append(f"Salaire : {salary}")
    if experience:
        lines.append(f"Expérience demandée : {experience}")
    if skills:
        lines.append(f"Compétences : {', '.join(skills[:10])}")
    lines.append(f"\nDescription :\n{body}")
    return "\n".join(lines)


class FranceTravailScraper(BaseScraper):
    """Scraper querying France Travail partner API with public web fallback.

    Attributes:
        departement: Default target department code (e.g. '13').
        detail_delay_seconds: Polite pause between public detail fetches.
        location_matcher: Shared geo matcher enforcing 25km radius in dept 13.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        departement: str = "13",
        detail_delay_seconds: float = 0.25,
        location_matcher: LocationMatcher | None = None,
    ) -> None:
        """Initialize France Travail scraper with optional OAuth2 credentials.

        Args:
            client_id: Client identifier from francetravail.io.
            client_secret: Client secret from francetravail.io.
            departement: French department code filter.
            detail_delay_seconds: Pause before each public detail HTTP call.
            location_matcher: Shared geo matcher for dept 13 filtering.
        """
        super().__init__(name="france_travail")
        settings = get_settings()

        self._client_id: str = (
            client_id
            if client_id is not None
            else settings.france_travail_client_id.get_secret_value()
        )
        self._client_secret: str = (
            client_secret
            if client_secret is not None
            else settings.france_travail_client_secret.get_secret_value()
        )
        self.departement: str = departement.strip() or "13"
        self.detail_delay_seconds: float = detail_delay_seconds
        self.location_matcher: LocationMatcher = (
            location_matcher or LocationMatcher()
        )

        # Gestion en mémoire du jeton OAuth2
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    async def _get_access_token(
        self, client: httpx.AsyncClient
    ) -> str | None:
        """Fetch or return cached OAuth2 access token.

        Args:
            client: Active HTTP client session.

        Returns:
            str | None: Valid bearer token, or None on authentication failure.
        """
        # Vérification si le jeton en cache est toujours valide (marge de 60s)
        now = time.monotonic()
        if self._access_token and now < (self._token_expires_at - 60):
            return self._access_token

        if not self._client_id or not self._client_secret:
            return None

        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": DEFAULT_OAUTH_SCOPE,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            # Demande d'un nouveau jeton d'accès
            response = await client.post(
                FRANCE_TRAVAIL_TOKEN_URL,
                data=data,
                headers=headers,
            )
            if response.status_code != 200:
                print(
                    f"⚠️ [FranceTravail] Rejet du jeton OAuth2 : "
                    f"HTTP {response.status_code}"
                )
                return None

            token_data = response.json()
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 1499)

            if isinstance(access_token, str):
                self._access_token = access_token
                self._token_expires_at = time.monotonic() + float(expires_in)
                return self._access_token

            return None

        except httpx.HTTPError as err:
            print(f"⚠️ [FranceTravail] Erreur réseau OAuth2 : {err}")
            return None

    @staticmethod
    def _is_remote_hint(title: str, location: str) -> bool:
        """Return True if offer title or location indicates remote work."""
        norm = f"{normalize_geo_text(title)} {normalize_geo_text(location)}"
        return any(
            tok in norm
            for tok in ("remote", "teletravail", "full remote", "distanciel")
        )

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch job postings via France Travail API v2 or public web fallback.

        Args:
            keywords: List of target job keywords.
            locations: Geographic focus terms (supplementary to departement).
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Normalized job post instance.
        """
        async with httpx.AsyncClient(
            headers=DEFAULT_PUBLIC_HEADERS,
            timeout=15.0,
            follow_redirects=True,
        ) as client:
            # 1. Voie officielle OAuth2 si identifiants configurés
            if self._client_id and self._client_secret:
                token = await self._get_access_token(client)
                if token:
                    api_yielded = 0
                    try:
                        async for raw in self._scrape_official_api(
                            client=client,
                            token=token,
                            keywords=keywords,
                            locations=locations,
                            limit=limit,
                        ):
                            api_yielded += 1
                            yield raw
                        return
                    except PermissionError as perm_err:
                        print(
                            f"⚠️ [FranceTravail] {perm_err} — bascule sur le "
                            "repli public candidat.francetravail.fr..."
                        )
                        if api_yielded > 0:
                            return
                else:
                    print(
                        "⚠️ [FranceTravail] Échec API OAuth2, bascule sur le "
                        "repli public candidat.francetravail.fr..."
                    )
            else:
                print(
                    "ℹ️ [FranceTravail] Clés API absentes : utilisation du "
                    "repli public candidat.francetravail.fr."
                )

            # 2. Repli public sans clé API (HTML recherche + JSON détail)
            async for raw in self._scrape_public_web(
                client=client,
                keywords=keywords,
                locations=locations,
                limit=limit,
            ):
                yield raw

    async def _scrape_official_api(
        self,
        client: httpx.AsyncClient,
        token: str,
        keywords: list[str],
        locations: list[str],
        limit: int,
    ) -> AsyncIterator[RawJobListing]:
        """Query the official France Travail Partner v2 REST API."""
        yielded_count = 0
        seen_ids: set[str] = set()
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        for keyword in keywords:
            if yielded_count >= limit:
                break

            batch_limit = min(max(limit - yielded_count, 10), 49)
            query_params: dict[str, Any] = {
                "motsCles": keyword,
                "departement": self.departement,
                "typeContrat": "CDI,CDD,MIS",
                "range": f"0-{batch_limit}",
            }

            try:
                response = await client.get(
                    FRANCE_TRAVAIL_SEARCH_URL,
                    headers=auth_headers,
                    params=query_params,
                )
                # 200 (complet), 206 (partiel), 204 (aucun résultat)
                if response.status_code in (401, 403):
                    raise PermissionError(
                        f"Accès API refusé (HTTP {response.status_code})"
                    )
                if response.status_code == 204:
                    continue
                if response.status_code not in (200, 206):
                    continue

                data = response.json()
                offers = data.get("resultats", [])

                for offer in offers:
                    if yielded_count >= limit:
                        break

                    offer_id = str(offer.get("id", "")).strip()
                    title = str(offer.get("intitule", "")).strip()
                    description = str(offer.get("description", "")).strip()
                    if not offer_id or not title or offer_id in seen_ids:
                        continue
                    if is_excluded_title(title):
                        continue

                    location_info = offer.get("lieuTravail") or {}
                    location_name = str(
                        location_info.get(
                            "libelle", f"Dép. {self.departement}"
                        )
                    ).strip()

                    if not self._is_remote_hint(title, location_name):
                        allowed = (
                            await self.location_matcher.is_location_allowed(
                                raw_location=location_name,
                                extra_locations=locations,
                                client=client,
                            )
                        )
                        if not allowed:
                            continue

                    seen_ids.add(offer_id)
                    company_info = offer.get("entreprise") or {}
                    company_name = str(
                        company_info.get("nom", "Entreprise confidentielle")
                    ).strip()

                    origin = offer.get("origineOffre") or {}
                    detail_url = f"{FT_PUBLIC_DETAIL_PAGE}/{offer_id}"
                    raw_url = str(origin.get("urlOrigine") or detail_url)

                    contract_type = str(
                        offer.get("typeContratLibelle", "")
                    ).strip()
                    full_description = (
                        f"{title} chez {company_name}\n"
                        f"Lieu : {location_name} | Contrat : {contract_type}\n"
                        f"\nDescription :\n{description}"
                    )

                    payload = dict(offer)
                    payload.setdefault("is_easy_apply", False)
                    payload.setdefault("apply_url", raw_url)

                    yield RawJobListing(
                        platform=self.name,
                        company=company_name,
                        title=title,
                        url=raw_url,
                        description=full_description,
                        location=location_name,
                        raw_payload=payload,
                    )
                    yielded_count += 1

            except httpx.HTTPError as err:
                print(
                    f"⚠️ [FranceTravail] Erreur réseau API pour "
                    f"'{keyword}': {err}"
                )
                continue

    async def _scrape_public_web(
        self,
        client: httpx.AsyncClient,
        keywords: list[str],
        locations: list[str],
        limit: int,
    ) -> AsyncIterator[RawJobListing]:
        """Scrape public candidat.francetravail.fr pages without API keys."""
        yielded_count = 0
        seen_ids: set[str] = set()
        dept_code = (
            f"{self.departement}D"
            if self.departement.isdigit()
            else self.departement
        )

        for keyword in keywords:
            if yielded_count >= limit:
                break

            params = {
                "motsCles": keyword,
                "lieux": dept_code,
                "typeContrat": "CDI,CDD,MIS",
                "offresPartenaires": "true",
            }
            try:
                res = await client.get(FT_PUBLIC_SEARCH_URL, params=params)
                if res.status_code != 200:
                    continue
            except httpx.HTTPError as err:
                print(
                    f"⚠️ [FranceTravail] Erreur recherche publique "
                    f"'{keyword}': {err}"
                )
                continue

            cards = re.findall(
                r'<li[^>]*class="result[^"]*"[^>]*>(.*?)</li>',
                res.text,
                re.DOTALL | re.IGNORECASE,
            )

            for card in cards:
                if yielded_count >= limit:
                    break

                parsed = self._parse_public_search_card(card)
                if parsed is None:
                    continue

                offer_id, title, company, loc, contract, snippet = parsed
                if offer_id in seen_ids or is_excluded_title(title):
                    continue

                if loc and not self._is_remote_hint(title, loc):
                    allowed = await self.location_matcher.is_location_allowed(
                        raw_location=loc,
                        extra_locations=locations,
                        client=client,
                    )
                    if not allowed:
                        continue

                seen_ids.add(offer_id)
                detail_url = f"{FT_PUBLIC_DETAIL_PAGE}/{offer_id}"

                # Pause de courtoisie avant récupération du JSON complet
                if self.detail_delay_seconds > 0:
                    await asyncio.sleep(self.detail_delay_seconds)

                detail_json = await self._fetch_public_detail_json(
                    client=client,
                    offer_id=offer_id,
                    referer=detail_url,
                )

                if detail_json:
                    etab = detail_json.get("etablissement") or {}
                    company = str(
                        etab.get("libelleEnseigne") or company
                    ).strip()
                    lieu_info = detail_json.get("lieuTravail") or {}
                    loc = str(
                        lieu_info.get("syntheseLieuTravail") or loc
                    ).strip()
                    full_description = _format_public_offer_description(
                        detail=detail_json,
                        fallback_title=title,
                        fallback_company=company,
                        fallback_location=loc,
                        fallback_contract=contract,
                        fallback_snippet=snippet,
                    )
                else:
                    full_description = (
                        f"{title} chez {company}\n"
                        f"Lieu : {loc} | Contrat : {contract}\n\n"
                        f"Description :\n{snippet or title}"
                    )

                yield RawJobListing(
                    platform=self.name,
                    company=company,
                    title=title,
                    url=detail_url,
                    description=full_description,
                    location=loc,
                    raw_payload={
                        "offer_id": offer_id,
                        "source": "public_web",
                        "is_easy_apply": False,
                        "apply_url": detail_url,
                    },
                )
                yielded_count += 1

    @staticmethod
    def _parse_public_search_card(
        card_html: str,
    ) -> tuple[str, str, str, str, str, str] | None:
        """Extract offer summary fields from a public search result card."""
        id_match = re.search(
            r'href=["\']/offres/recherche/detail/([A-Za-z0-9]+)["\']',
            card_html,
        )
        title_match = re.search(
            r'<span[^>]*class=["\']media-heading-title["\'][^>]*>'
            r"(.*?)</span>",
            card_html,
            re.DOTALL,
        )
        if not id_match or not title_match:
            return None

        offer_id = id_match.group(1).strip()
        title = _clean_html(title_match.group(1))
        if not offer_id or not title:
            return None

        company = "Entreprise confidentielle"
        loc = ""
        sub_match = re.search(
            r'<p[^>]*class=["\'][^"\']*subtext[^"\']*["\'][^>]*>(.*?)</p>',
            card_html,
            re.DOTALL,
        )
        if sub_match:
            sub_html = sub_match.group(1)
            loc_span = re.search(
                r"<span[^>]*>(.*?)</span>", sub_html, re.DOTALL
            )
            if loc_span:
                loc = _clean_html(loc_span.group(1))
                comp_part = sub_html[:loc_span.start()]
            else:
                comp_part = sub_html
            comp_clean = _clean_html(comp_part).rstrip("- \xa0").strip()
            if comp_clean:
                company = comp_clean

        contract_match = re.search(
            r'<p[^>]*class=["\']contrat[^"\']*["\'][^>]*>(.*?)</p>',
            card_html,
            re.DOTALL,
        )
        contract = _clean_html(contract_match.group(1)) if contract_match else ""

        desc_match = re.search(
            r'<p[^>]*class=["\']description["\'][^>]*>(.*?)</p>',
            card_html,
            re.DOTALL,
        )
        snippet = _clean_html(desc_match.group(1)) if desc_match else ""

        return offer_id, title, company, loc, contract, snippet

    @staticmethod
    async def _fetch_public_detail_json(
        client: httpx.AsyncClient,
        offer_id: str,
        referer: str,
    ) -> dict[str, Any] | None:
        """Fetch full structured offer details from public JSON endpoint."""
        url = f"{FT_PUBLIC_DETAIL_API}/{offer_id}"
        headers = {
            "Accept": "application/json",
            "Referer": referer,
        }
        try:
            res = await client.get(url, headers=headers)
            if res.status_code != 200:
                return None
            data = res.json()
            return data if isinstance(data, dict) else None
        except (httpx.HTTPError, ValueError):
            return None
