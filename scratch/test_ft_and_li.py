"""Live verification for FranceTravail public fallback and LinkedInScraper."""

import asyncio

from scrapers.france_travail import FranceTravailScraper
from scrapers.linkedin import LinkedInScraper


async def main() -> None:
    """Run targeted live tests for France Travail fallback and LinkedIn."""
    ft_scraper = FranceTravailScraper(client_id="", client_secret="")
    ft_jobs = [
        job
        async for job in ft_scraper.scrape(
            keywords=["Python", "DevSecOps"],
            locations=["Marseille", "Aix-en-Provence", "13", "Télétravail"],
            limit=3,
        )
    ]
    print(f"FranceTravail public fallback count: {len(ft_jobs)}")
    assert len(ft_jobs) > 0, "Expected >= 1 France Travail public job"
    for job in ft_jobs:
        print(
            f"  [FT] {job.company} | {job.title} | {job.location} | "
            f"len={len(job.description)} | easy={job.raw_payload.get('is_easy_apply')}"
        )
        assert len(job.description) > 200
        assert job.raw_payload.get("is_easy_apply") is False

    li_scraper = LinkedInScraper()
    li_jobs = [
        job
        async for job in li_scraper.scrape(
            keywords=["Python", "Cybersécurité"],
            locations=["Bouches-du-Rhône", "Marseille"],
            limit=3,
        )
    ]
    print(f"LinkedIn count: {len(li_jobs)}")
    assert len(li_jobs) > 0, "Expected >= 1 LinkedIn job"
    for job in li_jobs:
        print(
            f"  [LI] {job.company} | {job.title} | {job.location} | "
            f"len={len(job.description)} | easy={job.raw_payload.get('is_easy_apply')}"
        )
        assert len(job.description) > 150


if __name__ == "__main__":
    asyncio.run(main())
