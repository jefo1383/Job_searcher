"""Scraper targeting CMA CGM career portal via SuccessFactors sitemap & HTML."""

import asyncio
from collections.abc import AsyncIterator
import html
import re
from typing import Any
import unicodedata
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import httpx

from core.location_matcher import LocationMatcher
from scrapers.base import BaseScraper, is_excluded_title
from scrapers.models import RawJobListing

# Points d'accès SuccessFactors officiels de CMA CGM
CMA_CGM_SITEMAP_URL: str = "https://jobs.cmacgm-group.com/sitemap.xml"
CMA_CGM_RSS_BASE_URL: str = (
    "https://jobs.cmacgm-group.com/services/rss/job/?locale=fr_FR"
)

# Paramètres réglables : pré-filtrage géographique, tokens IT bilingues et débit
DEFAULT_CMA_LOCAL_HINTS: tuple[str, ...] = (
    "marseille",
    "aix",
    "aubagne",
    "marignane",
    "vitrolles",
    "ciotat",
    "allauch",
    "bouches",
    "fos",
    "martigues",
    "salon",
    "remote",
    "teletravail",
)

DEFAULT_CMA_IT_TOKENS: tuple[str, ...] = (
    "ia",
    "ai",
    "intelligence artificielle",
    "data",
    "python",
    "dev",
    "developer",
    "developpeur",
    "software",
    "logiciel",
    "cyber",
    "sec",
    "devops",
    "devsecops",
    "soc",
    "csirt",
    "cloud",
    "system",
    "systeme",
    "reseau",
    "network",
    "digital",
    "digitale",
    "informatique",
    "r&d",
    "tech",
    "ml",
    "llm",
    "analytics",
    "automation",
    "automatisation",
    "application",
    "applications",
    "backend",
    "fullstack",
    "sre",
    "it",
)

DEFAULT_CMA_DETAIL_DELAY_SECONDS: float = 0.15
DEFAULT_CMA_MAX_DETAIL_FETCHES: int = 40

DEFAULT_CMA_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


def _strip_accents_lower(text: str) -> str:
    """Normalize accents and lowercase text for token matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).lower()


def _clean_html_text(raw_text: str | None) -> str:
    """Decode HTML entities and strip markup tags into normalized plain text.

    Args:
        raw_text: Raw string possibly containing HTML markup.

    Returns:
        str: Cleaned text without HTML markup.
    """
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", stripped).strip()


def _extract_balanced_tag_at(
    html_text: str,
    start_pos: int,
    tag_name: str,
) -> str:
    """Extract inner HTML from start_pos up to its matching closing tag."""
    tag_regex = re.compile(
        rf"<(/?){re.escape(tag_name)}\b[^>]*>",
        re.IGNORECASE,
    )
    depth = 1
    for tag_match in tag_regex.finditer(html_text, pos=start_pos):
        if tag_match.group(1) == "/":
            depth -= 1
            if depth == 0:
                return html_text[start_pos : tag_match.start()]
        else:
            depth += 1
    return ""


def _extract_balanced_tag(
    html_text: str,
    open_pattern: str,
    tag_name: str,
) -> str:
    """Extract inner HTML of the first element matching open_pattern."""
    match = re.search(open_pattern, html_text, re.IGNORECASE)
    if not match:
        return ""
    return _extract_balanced_tag_at(html_text, match.end(), tag_name)


def _extract_all_description_blocks(html_text: str) -> str:
    """Extract and join all job description blocks from SuccessFactors HTML."""
    pattern = re.compile(
        r'<span[^>]*itemprop=["\']description["\'][^>]*>',
        re.IGNORECASE,
    )
    blocks: list[str] = []
    for match in pattern.finditer(html_text):
        inner = _extract_balanced_tag_at(html_text, match.end(), "span")
        cleaned = _clean_html_text(inner)
        if not cleaned:
            continue
        # Ignorer les blocs contenant uniquement du CSS inline ou très courts
        if cleaned.startswith(".") and "{" in cleaned[:120]:
            continue
        blocks.append(cleaned)
    return "\n\n".join(blocks).strip()


def _parse_slug_metadata(job_url: str) -> tuple[str, str]:
    """Extract city hint and readable title from a SuccessFactors job URL."""
    parts = [part for part in job_url.split("/") if part]
    slug = ""
    for idx, part in enumerate(parts):
        if part.lower() == "job" and idx + 1 < len(parts):
            slug = unquote(parts[idx + 1])
            break
    if not slug:
        return "", ""

    words = slug.replace("-", " ").strip()
    first_token = slug.split("-", 1)[0].strip()
    if slug.lower().startswith("aix-en-provence"):
        first_token = "Aix-en-Provence"
    elif slug.lower().startswith("la-ciotat"):
        first_token = "La Ciotat"
    return first_token, re.sub(r"\s+", " ", words)


class CMACGMScraper(BaseScraper):
    """Scraper extracting job postings from CMA CGM via sitemap & RSS fallback.

    Attributes:
        sitemap_url: Primary SuccessFactors XML sitemap listing all open jobs.
        base_rss_url: Fallback RSS syndication feed URL.
        location_matcher: Shared geographical filter with 25 km radius check.
        max_detail_fetches: Safety cap on detail page downloads per run.
    """

    def __init__(
        self,
        base_rss_url: str = CMA_CGM_RSS_BASE_URL,
        sitemap_url: str = CMA_CGM_SITEMAP_URL,
        location_matcher: LocationMatcher | None = None,
        max_detail_fetches: int = DEFAULT_CMA_MAX_DETAIL_FETCHES,
    ) -> None:
        """Initialize the CMA CGM scraper.

        Args:
            base_rss_url: Fallback endpoint URL for the RSS syndication feed.
            sitemap_url: XML sitemap URL containing all active CMA CGM offers.
            location_matcher: Shared geo matcher with 25 km distance check.
            max_detail_fetches: Max detail HTML requests executed per cycle.
        """
        super().__init__(name="cma_cgm")
        self.base_rss_url = base_rss_url
        self.sitemap_url = sitemap_url
        self.location_matcher = location_matcher or LocationMatcher()
        self.max_detail_fetches = max_detail_fetches

    @staticmethod
    def _matches_keywords_or_it(
        text: str,
        keywords: list[str],
    ) -> bool:
        """Return True when text matches explicit keywords or bilingual IT tokens."""
        norm_text = _strip_accents_lower(text)
        for kw in keywords:
            norm_kw = _strip_accents_lower(kw).strip()
            if norm_kw and norm_kw in norm_text:
                return True

        for token in DEFAULT_CMA_IT_TOKENS:
            escaped = re.escape(token)
            if token == "r&d":
                if re.search(
                    r"(?<![a-z0-9])r(?:\s*&\s*|\s+)d(?![a-z0-9])",
                    norm_text,
                ):
                    return True
                continue
            pattern = (
                rf"(?<![a-z0-9]){escaped}"
                if token.endswith("developp")
                else rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
            )
            if re.search(pattern, norm_text):
                return True
        return False

    @staticmethod
    def _has_local_hint(slug_text: str, locations: list[str]) -> bool:
        """Fast pre-filter avoiding GeoAPI calls on global CEVA/CMA URLs."""
        norm_slug = _strip_accents_lower(slug_text)
        hints = set(DEFAULT_CMA_LOCAL_HINTS)
        for loc in locations:
            cleaned = _strip_accents_lower(loc).strip()
            if cleaned and len(cleaned) >= 3:
                hints.add(cleaned)
        return any(hint in norm_slug for hint in hints)

    async def _scrape_from_sitemap(
        self,
        client: httpx.AsyncClient,
        keywords: list[str],
        locations: list[str],
        limit: int,
    ) -> AsyncIterator[RawJobListing]:
        """Discover local IT jobs via sitemap.xml and parse detail HTML."""
        response = await client.get(self.sitemap_url)
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} from sitemap",
                request=response.request,
                response=response,
            )

        raw_urls = re.findall(r"<loc>(.*?)</loc>", response.text)
        seen_urls: set[str] = set()
        candidates: list[tuple[str, str, str]] = []

        for raw_url in raw_urls:
            job_url = html.unescape(raw_url).strip()
            if "/job/" not in job_url or job_url in seen_urls:
                continue
            seen_urls.add(job_url)

            city_hint, slug_title = _parse_slug_metadata(job_url)
            if not slug_title:
                continue

            # Pré-filtre rapide localité + séniorité + mots-clés/IT sur le slug
            if not self._has_local_hint(f"{city_hint} {slug_title}", locations):
                continue
            if is_excluded_title(slug_title):
                continue
            if not self._matches_keywords_or_it(slug_title, keywords):
                continue

            candidates.append((job_url, city_hint, slug_title))

        yielded_count = 0
        fetched_count = 0
        for job_url, city_hint, slug_title in candidates:
            if yielded_count >= limit or fetched_count >= self.max_detail_fetches:
                break

            fetched_count += 1
            if fetched_count > 1:
                await asyncio.sleep(DEFAULT_CMA_DETAIL_DELAY_SECONDS)

            try:
                detail_res = await client.get(job_url)
                if detail_res.status_code != 200:
                    continue
            except httpx.HTTPError:
                continue

            page_html = detail_res.text
            raw_title_html = _extract_balanced_tag(
                page_html,
                r'<span[^>]*itemprop=["\']title["\'][^>]*>',
                tag_name="span",
            )
            title = _clean_html_text(raw_title_html) or slug_title
            if not title or is_excluded_title(title):
                continue

            location_text = city_hint or "Marseille"
            if locations and not await self.location_matcher.is_location_allowed(
                raw_location=location_text,
                extra_locations=locations,
                client=client,
            ):
                continue

            cleaned_desc = _extract_all_description_blocks(page_html)
            if not cleaned_desc:
                cleaned_desc = title

            full_text = f"{title} {cleaned_desc}"
            if not self._matches_keywords_or_it(full_text, keywords):
                continue

            composed_description = (
                f"{title} chez CMA CGM\n"
                f"Lieu : {location_text}\n\n"
                f"Description de l'offre :\n{cleaned_desc}"
            )
            raw_payload: dict[str, Any] = {
                "location": location_text,
                "is_easy_apply": False,
                "apply_url": job_url,
                "source": "sitemap",
            }

            yield RawJobListing(
                platform=self.name,
                company="CMA CGM",
                title=title,
                url=job_url,
                description=composed_description,
                location=location_text,
                raw_payload=raw_payload,
            )
            yielded_count += 1

    async def _scrape_from_rss_fallback(
        self,
        client: httpx.AsyncClient,
        keywords: list[str],
        locations: list[str],
        limit: int,
    ) -> AsyncIterator[RawJobListing]:
        """Fallback to SuccessFactors RSS feed if sitemap fails."""
        response = await client.get(self.base_rss_url)
        if response.status_code != 200:
            print(
                f"⚠️ [CMACGMScraper] HTTP {response.status_code} "
                f"from {self.base_rss_url}"
            )
            return

        root = ET.fromstring(response.text)
        channel = root.find("channel")
        if channel is None:
            return

        yielded_count = 0
        for item in channel.findall("item"):
            if yielded_count >= limit:
                break

            raw_title = item.findtext("title") or ""
            raw_link = item.findtext("link") or ""
            raw_desc = item.findtext("description") or ""
            raw_cat = item.findtext("category") or ""

            if not raw_title or not raw_link or is_excluded_title(raw_title):
                continue

            cleaned_desc = _clean_html_text(raw_desc)
            full_search_text = f"{raw_title} {cleaned_desc} {raw_cat}"
            city_hint, _ = _parse_slug_metadata(raw_link)
            location_matched = city_hint or "Marseille"
            if locations and not await self.location_matcher.is_location_allowed(
                raw_location=location_matched,
                extra_locations=locations,
                client=client,
            ):
                continue

            if not self._matches_keywords_or_it(full_search_text, keywords):
                continue

            raw_payload: dict[str, Any] = {
                "category": raw_cat,
                "raw_title": raw_title,
                "location": location_matched,
                "is_easy_apply": False,
                "apply_url": raw_link.strip(),
                "source": "rss",
            }
            composed_description = (
                f"{raw_title.strip()} chez CMA CGM\n"
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
        try:
            async with httpx.AsyncClient(
                headers=DEFAULT_CMA_HEADERS,
                timeout=15.0,
                follow_redirects=True,
            ) as client:
                try:
                    async for listing in self._scrape_from_sitemap(
                        client=client,
                        keywords=keywords,
                        locations=locations,
                        limit=limit,
                    ):
                        yield listing
                    return
                except httpx.HTTPError as sitemap_err:
                    print(
                        f"⚠️ [CMACGMScraper] Sitemap indisponible "
                        f"({sitemap_err}), repli RSS..."
                    )

                async for listing in self._scrape_from_rss_fallback(
                    client=client,
                    keywords=keywords,
                    locations=locations,
                    limit=limit,
                ):
                    yield listing

        except httpx.HTTPError as err:
            print(f"⚠️ [CMACGMScraper] Network error fetching feed: {err}")
        except ET.ParseError as err:
            print(f"⚠️ [CMACGMScraper] XML parsing error: {err}")
