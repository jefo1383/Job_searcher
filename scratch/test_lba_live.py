"""Live verification for LaBonneAlternanceScraper with real LBA_API_KEY."""

import asyncio

from config.criteria import parse_search_criteria_file
from config.settings import get_settings
from scrapers.la_bonne_alternance import LaBonneAlternanceScraper


async def main() -> None:
    """Query live La bonne alternance API and verify IT/location filtering."""
    settings = get_settings()
    criteria = parse_search_criteria_file(settings.search_criteria_path)
    scraper = LaBonneAlternanceScraper()
    jobs = [
        job
        async for job in scraper.scrape(
            keywords=criteria.keywords,
            locations=criteria.locations,
            limit=10,
        )
    ]
    print(f"LBA IT alternance jobs count: {len(jobs)}")
    assert len(jobs) > 0, "Expected >= 1 IT alternance offer from LBA API"
    for job in jobs:
        print(
            f"  [LBA:{job.raw_payload.get('partner_label')}] "
            f"{job.company} | {job.title} | {job.location} | "
            f"len={len(job.description)} | rome={job.raw_payload.get('rome_codes')} | "
            f"easy={job.raw_payload.get('is_easy_apply')}"
        )
        assert job.raw_payload.get("partner_label") != "recruteurs_lba"
        assert job.raw_payload.get("is_easy_apply") is False
        assert len(job.description) > 100


if __name__ == "__main__":
    asyncio.run(main())
