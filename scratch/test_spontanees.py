"""Live verification for La bonne alternance /spontanees recruiters."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from telegram.ext import CommandHandler

from db.database import JobDatabase
from hitl.telegram_bot import JobHunterBot
from scrapers.la_bonne_alternance import LaBonneAlternanceScraper


async def main() -> None:
    """Verify IT recruiters fetch, distance sorting, and Telegram formatting."""
    scraper = LaBonneAlternanceScraper()
    recs = await scraper.fetch_spontaneous_recruiters(it_only=True)
    print(f"IT spontaneous recruiters within 25km: {len(recs)}")
    assert len(recs) > 0, "Expected >= 1 IT recruiter within 25km"
    for rec in recs[:5]:
        print(
            f"  {rec.company} | {rec.distance_km} km | {rec.naf_code} "
            f"{rec.naf_label} | {rec.address}"
        )
        assert rec.distance_km is None or rec.distance_km <= 25.0

    with TemporaryDirectory() as tmp:
        db = JobDatabase(Path(tmp) / "test.db")
        await db.init_db()
        bot = JobHunterBot(
            db=db,
            bot_token="123456:DUMMY_TOKEN",
            chat_id=12345,
            lba_scraper=scraper,
        )
        assert any(
            isinstance(h, CommandHandler) and "spontanees" in h.commands
            for group in bot.application.handlers.values()
            for h in group
        ), "Expected /spontanees CommandHandler registered"

        msg, markup = bot._build_spontanees_page(recs, offset=0, page_size=8)
        assert len(msg) < 4096
        assert markup is not None
        print(f"Formatted Telegram message length: {len(msg)} chars")


if __name__ == "__main__":
    asyncio.run(main())
