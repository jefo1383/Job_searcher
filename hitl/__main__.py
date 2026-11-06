"""CLI entrypoint for running Job Hunter Telegram bot."""

from config.settings import get_settings
from db.database import JobDatabase
from hitl.telegram_bot import JobHunterBot


def main() -> None:
    """Initialize database and start bot in polling mode."""
    settings = get_settings()
    database = JobDatabase(settings.database_path)
    bot = JobHunterBot(db=database)
    bot.run_polling()


if __name__ == "__main__":
    main()
