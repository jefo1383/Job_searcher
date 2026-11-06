"""Scrapers package providing unified job collection and deduplication."""

from scrapers.base import BaseScraper
from scrapers.cma_cgm import CMACGMScraper
from scrapers.engine import ScraperEngine
from scrapers.france_travail import FranceTravailScraper
from scrapers.hellowork import HelloWorkScraper
from scrapers.indeed import IndeedScraper
from scrapers.models import RawJobListing
from scrapers.wttj import WTTJScraper

__all__ = [
    "BaseScraper",
    "CMACGMScraper",
    "FranceTravailScraper",
    "HelloWorkScraper",
    "IndeedScraper",
    "RawJobListing",
    "ScraperEngine",
    "WTTJScraper",
]
