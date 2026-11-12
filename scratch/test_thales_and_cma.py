"""Live integration check for ThalesScraper (Phenom) and CMACGMScraper."""

import asyncio
from pathlib import Path

from config.criteria import parse_search_criteria_file
from scrapers.cma_cgm import CMACGMScraper
from scrapers.thales import ThalesScraper


async def main() -> None:
    """Verify both fixed scrapers against Profile/search_criteria.txt."""
    criteria = parse_search_criteria_file(Path("Profile/search_criteria.txt"))

    print("=== Testing ThalesScraper (Phenom SSR) ===")
    thales = ThalesScraper()
    thales_jobs = []
    async for job in thales.scrape(
        keywords=criteria.keywords,
        locations=criteria.locations,
        limit=10,
    ):
        thales_jobs.append(job)
        has_apply = bool(job.raw_payload and job.raw_payload.get("apply_url"))
        print(
            f"- [{job.platform}] {job.title} | {job.location} | "
            f"len={len(job.description)} | apply={has_apply}"
        )
    print(f"Thales total collected: {len(thales_jobs)}")
    assert len(thales_jobs) > 0, "Expected >= 1 Thales offer"
    assert all(len(j.description) > 300 for j in thales_jobs)

    print("\n=== Testing CMACGMScraper (Sitemap + HTML) ===")
    cma = CMACGMScraper()
    cma_jobs = []
    async for job in cma.scrape(
        keywords=criteria.keywords,
        locations=criteria.locations,
        limit=10,
    ):
        cma_jobs.append(job)
        print(
            f"- [{job.platform}] {job.title} | {job.location} | "
            f"len={len(job.description)} | {job.url}"
        )
    print(f"CMA CGM total collected: {len(cma_jobs)}")
    assert len(cma_jobs) > 0, "Expected >= 1 CMA CGM offer"
    assert all(len(j.description) > 300 for j in cma_jobs)


if __name__ == "__main__":
    asyncio.run(main())
