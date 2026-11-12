"""Unit and DOM smoke test for Indeed Choice 1 and Telegram /indeed wiring."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.async_api import async_playwright
from telegram.ext import CommandHandler

from core.location_matcher import LocationMatcher
from db.database import JobDatabase
from hitl.telegram_bot import JobHunterBot
from main import build_arg_parser
from scrapers.indeed import IndeedScraper

MOCK_INDEED_HTML = """
<!doctype html>
<html>
<head><title>Emplois Python - Bouches-du-Rhône | Indeed</title></head>
<body>
  <div class="job_seen_beacon" data-jk="abc12345">
    <a class="jcs-JobTitle" data-jk="abc12345"
       href="https://fr.indeed.com/viewjob?jk=abc12345"
       onclick="document.getElementById('jobDescriptionText').textContent =
       'Nous recherchons un Développeur Python / DevSecOps junior sur ' +
       'Marseille avec expérience FastAPI, Docker et PostgreSQL.'; return false;">
      Développeur Python / DevSecOps (H/F)
    </a>
    <span data-testid="company-name">Acme Cyber</span>
    <div data-testid="text-location">Marseille (13)</div>
    <div class="job-snippet">Poste backend Python...</div>
  </div>
  <div class="job_seen_beacon" data-jk="senior999">
    <a class="jcs-JobTitle" data-jk="senior999"
       href="https://fr.indeed.com/viewjob?jk=senior999">
      Senior Engineering Manager (H/F)
    </a>
    <span data-testid="company-name">BigCorp</span>
    <div data-testid="text-location">Marseille (13)</div>
  </div>
  <div id="jobDescriptionText" style="min-height: 20px;"></div>
</body>
</html>
"""


async def main() -> None:
    """Verify CLI parser, bot command handler, and DOM extraction."""
    parser = build_arg_parser()
    args = parser.parse_args(["--indeed", "--limit", "5"])
    assert args.indeed is True
    assert args.limit == 5

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = JobDatabase(tmp_path / "test.db")
        await db.init_db()
        bot = JobHunterBot(
            db=db,
            bot_token="123456:DUMMY_TOKEN",
            chat_id=12345,
        )
        assert any(
            isinstance(h, CommandHandler) and "indeed" in h.commands
            for group in bot.application.handlers.values()
            for h in group
        ), "Expected /indeed CommandHandler registered"

        matcher = LocationMatcher(
            criteria_locations=["Marseille", "13"],
            cache_path=tmp_path / "loc_cache.json",
            api_key="DUMMY_KEY",
        )
        scraper = IndeedScraper(
            user_data_dir=tmp_path / "chrome_profile",
            headless=True,
            location_matcher=matcher,
        )
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_content(MOCK_INDEED_HTML)
            seen_jks: set[str] = set()
            items = await scraper._extract_cards_from_page(
                page=page,
                keyword="Python",
                search_loc="Marseille",
                allowed_locations=["Marseille", "13"],
                seen_jks=seen_jks,
                remaining_limit=10,
            )
            await browser.close()

        assert len(items) == 1, f"Expected 1 filtered job, got {len(items)}"
        job = items[0]
        assert job.company == "Acme Cyber"
        assert job.raw_payload.get("is_easy_apply") is False
        assert "FastAPI, Docker et PostgreSQL" in job.description
        print(
            f"Verified Indeed extraction: {job.company} | "
            f"len={len(job.description)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
