"""Scraper implementation for France Travail using official v2 API."""

from collections.abc import AsyncIterator
import time
from typing import Any
import httpx

from config.settings import get_settings
from scrapers.base import BaseScraper
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


class FranceTravailScraper(BaseScraper):
    """Scraper querying France Travail partner API with OAuth2 authentication.

    Attributes:
        departement: Default target department code (e.g. '13').
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        departement: str = "13",
    ) -> None:
        """Initialize France Travail scraper with OAuth2 credentials.

        Args:
            client_id: Client identifier from francetravail.io.
            client_secret: Client secret from francetravail.io.
            departement: French department code filter.
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
        self.departement: str = departement

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
                    f"⚠️ [FranceTravail] Token request rejected: "
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
            print(f"⚠️ [FranceTravail] Token network error: {err}")
            return None

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch job postings from France Travail API matching keywords.

        Args:
            keywords: List of target job keywords.
            locations: Geographic focus terms (supplementary to departement).
            limit: Maximum count of postings to extract.

        Yields:
            RawJobListing: Normalized job post instance.
        """
        if not self._client_id or not self._client_secret:
            print(
                "ℹ️ [FranceTravail] Identifiants API non configurés "
                "(FRANCE_TRAVAIL_CLIENT_ID / SECRET dans .env). "
                "Scraper ignoré pour cette session."
            )
            return

        yielded_count = 0
        async with httpx.AsyncClient(timeout=15.0) as client:
            token = await self._get_access_token(client)
            if not token:
                return

            auth_headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }

            for keyword in keywords:
                if yielded_count >= limit:
                    break

                batch_limit = min(limit - yielded_count, 49)
                query_params: dict[str, Any] = {
                    "motsCles": keyword,
                    "departement": self.departement,
                    # Contrats ciblés par votre profil (CDI, CDD, Intérim/Mission)
                    "typeContrat": "CDI,CDD,MIS",
                    "range": f"0-{batch_limit}",
                }

                try:
                    # Requête de recherche d'offres
                    response = await client.get(
                        FRANCE_TRAVAIL_SEARCH_URL,
                        headers=auth_headers,
                        params=query_params,
                    )

                    # France Travail renvoie :
                    # 200 (complet), 206 (partiel), 204 (aucun résultat)
                    if response.status_code == 204:
                        continue
                    if response.status_code not in (200, 206):
                        continue

                    data = response.json()
                    offers = data.get("resultats", [])

                    for offer in offers:
                        if yielded_count >= limit:
                            break

                        offer_id = offer.get("id", "")
                        title = offer.get("intitule", "")
                        description = offer.get("description", "")
                        if not offer_id or not title:
                            continue

                        company_info = offer.get("entreprise", {})
                        company_name = company_info.get(
                            "nom", "Entreprise confidentielle"
                        )

                        origin = offer.get("origineOffre", {})
                        raw_url = origin.get(
                            "urlOrigine",
                            f"https://candidat.francetravail.fr/offres/"
                            f"recherche/detail/{offer_id}",
                        )

                        location_info = offer.get("lieuTravail", {})
                        location_name = location_info.get(
                            "libelle", f"Dép. {self.departement}"
                        )

                        contract_type = offer.get("typeContratLibelle", "")
                        full_description = (
                            f"{title} chez {company_name}\n"
                            f"Lieu : {location_name} | Contrat : {contract_type}\n\n"
                            f"Description :\n{description}"
                        )

                        yield RawJobListing(
                            platform=self.name,
                            company=company_name,
                            title=title,
                            url=raw_url,
                            description=full_description,
                            location=location_name,
                            raw_payload=offer,
                        )
                        yielded_count += 1

                except httpx.HTTPError as err:
                    print(
                        f"⚠️ [FranceTravail] Search network warning for "
                        f"'{keyword}': {err}"
                    )
                    continue
