"""Scraper targeting CMA CGM career portal via SuccessFactors feed."""

from collections.abc import AsyncIterator
import re
from typing import Any
import xml.etree.ElementTree as ET
import httpx

from scrapers.base import BaseScraper
from scrapers.models import RawJobListing

# Flux RSS officiel SuccessFactors de CMA CGM
CMA_CGM_RSS_BASE_URL: str = (
    "https://jobs.cmacgm-group.com/services/rss/job/?locale=fr_FR"
)


def _clean_html_text(raw_text: str | None) -> str:
    """Strip HTML tags and condense whitespace.

    Args:
        raw_text: Raw string possibly containing HTML markup.

    Returns:
        str: Cleaned text without HTML markup.
    """
    if not raw_text:
        return ""
    stripped = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", stripped).strip()


class CMACGMScraper(BaseScraper):
    """Scraper extracting job postings directly from CMA CGM career syndication.

    Attributes:
        base_rss_url: URL to the SuccessFactors RSS feed.
    """

    def __init__(self, base_rss_url: str = CMA_CGM_RSS_BASE_URL) -> None:
        """Initialize the CMA CGM scraper.

        Args:
            base_rss_url: Endpoint URL for the RSS job syndication feed.
        """
        super().__init__(name="cma_cgm")
        self.base_rss_url = base_rss_url

    async def scrape(
        self,
        keywords: list[str],
        locations: list[str],
        limit: int = 20,
    ) -> AsyncIterator[RawJobListing]:
        """Fetch and filter CMA CGM job listings matching keywords and location.

        Args:
            keywords: List of target job keywords (e.g. 'IA', 'Python', 'SOC').
            locations: Geographic filters (e.g. 'Marseille', 'Remote').
            limit: Maximum count of matching postings to return.

        Yields:
            RawJobListing: Formatted job posting instance.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml",
        }

        yielded_count = 0
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Récupération du flux XML syndiqué
                response = await client.get(self.base_rss_url, headers=headers)
                if response.status_code != 200:
                    print(
                        f"⚠️ [CMACGMScraper] HTTP {response.status_code} "
                        f"from {self.base_rss_url}"
                    )
                    return

                # Parsing du flux XML avec le module standard ElementTree
                root = ET.fromstring(response.text)
                channel = root.find("channel")
                if channel is None:
                    return

                items = channel.findall("item")

                for item in items:
                    if yielded_count >= limit:
                        break

                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    desc_elem = item.find("description")
                    category_elem = item.find("category")

                    raw_title = title_elem.text if title_elem is not None else ""
                    raw_link = link_elem.text if link_elem is not None else ""
                    raw_desc = desc_elem.text if desc_elem is not None else ""
                    raw_cat = category_elem.text if category_elem is not None else ""

                    if not raw_title or not raw_link:
                        continue

                    cleaned_desc = _clean_html_text(raw_desc)
                    full_search_text = (
                        f"{raw_title} {cleaned_desc} {raw_cat}".lower()
                    )

                    # Filtrage géographique (Marseille / Bouches-du-Rhône / Remote)
                    location_matched = "Marseille / Non spécifié"
                    if locations:
                        matched_loc_found = False
                        for loc in locations:
                            if loc.lower() in full_search_text:
                                matched_loc_found = True
                                location_matched = loc
                                break
                        # Si aucune mention explicite de la localisation cible
                        if not matched_loc_found:
                            continue

                    # Filtrage par mots-clés techniques ou de métier
                    if keywords:
                        keyword_matched = any(
                            kw.lower() in full_search_text for kw in keywords
                        )
                        if not keyword_matched:
                            continue

                    raw_payload: dict[str, Any] = {
                        "category": raw_cat,
                        "raw_title": raw_title,
                    }

                    composed_description = (
                        f"{raw_title} chez CMA CGM\n"
                        f"Catégorie : {raw_cat or 'IT / Digital / Ingénierie'}\n"
                        f"Lieu : {location_matched}\n\n"
                        f"Description de l'offre :\n{cleaned_desc}"
                    )

                    yield RawJobListing(
                        platform=self.name,
                        company="CMA CGM",
                        title=raw_title.strip(),
                        url=raw_link.strip(),
                        description=composed_description,
                        location=location_matched,
                        raw_payload=raw_payload,
                    )
                    yielded_count += 1

        except httpx.HTTPError as err:
            print(f"⚠️ [CMACGMScraper] Network error fetching feed: {err}")
            return
        except ET.ParseError as err:
            print(f"⚠️ [CMACGMScraper] XML parsing error: {err}")
            return
