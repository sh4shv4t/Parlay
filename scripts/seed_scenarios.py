"""
Seed initial data into Parlay database.
Adds sample leaderboard entries for demo purposes.

Usage:
    python -m scripts.seed_scenarios
"""
import asyncio
import logging

import aiosqlite

logger = logging.getLogger(__name__)
DB_PATH = "parlay.db"

SAMPLE_ENTRIES = [
    ("ParlaBot-Alpha", "saas_enterprise",        "shark",    287.4, 0.82, 2, 1),
    ("NashSeeker",     "consulting_retainer",     "diplomat", 245.1, 0.75, 2, 1),
    ("AnchorMaster",   "hiring_package",          "analyst",  198.7, 0.68, 1, 1),
    ("ZOPARunner",     "vendor_hardware",         "veteran",  312.9, 0.91, 3, 1),
    ("DealDragon",     "acquisition_term_sheet",  "wildcard", 176.3, 0.55, 2, 1),
    ("SilentCloser",   "saas_enterprise",         "veteran",  298.6, 0.88, 2, 1),
    ("BluffCatcher",   "consulting_retainer",     "shark",    221.5, 0.72, 1, 1),
    ("Rubinstein99",   "hiring_package",          "wildcard", 189.2, 0.63, 2, 1),
]


async def seed(db_path: str = DB_PATH) -> None:
    """Insert sample leaderboard entries for demo."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM leaderboard")
        row = await cursor.fetchone()
        if row and row[0] > 0:
            logger.info("Leaderboard already has data — skipping seed")
            return

        await db.executemany(
            """
            INSERT INTO leaderboard
                (player_name, scenario_id, persona, total_reward,
                 deal_efficiency, acts_completed, deal_closed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            SAMPLE_ENTRIES,
        )
        await db.commit()
    logger.info(f"Seeded {len(SAMPLE_ENTRIES)} leaderboard entries")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed())
