"""
Initialize Parlay SQLite database tables.
Safe to run multiple times (CREATE TABLE IF NOT EXISTS).

Usage:
    python -m scripts.init_db
    # or imported by main.py lifespan
"""
import asyncio
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "parlay.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL UNIQUE,
    player_name   TEXT    NOT NULL,
    scenario_id   TEXT    NOT NULL,
    persona       TEXT    NOT NULL,
    started_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at  DATETIME,
    status        TEXT    DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id
    ON sessions(session_id);

CREATE TABLE IF NOT EXISTS leaderboard (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name   TEXT    NOT NULL,
    scenario_id   TEXT    NOT NULL,
    persona       TEXT    NOT NULL,
    total_reward  REAL    NOT NULL,
    deal_efficiency REAL  NOT NULL DEFAULT 0.0,
    acts_completed INTEGER NOT NULL DEFAULT 1,
    deal_closed   INTEGER NOT NULL DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_leaderboard_scenario
    ON leaderboard(scenario_id, total_reward DESC);

CREATE INDEX IF NOT EXISTS idx_leaderboard_global
    ON leaderboard(total_reward DESC);

CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL UNIQUE,
    player_name   TEXT    NOT NULL,
    scenario_id   TEXT    NOT NULL,
    persona       TEXT    NOT NULL,
    total_reward  REAL,
    deal_efficiency REAL,
    acts_completed INTEGER,
    deal_closed   INTEGER DEFAULT 0,
    turns         INTEGER DEFAULT 0,
    drift_adapted INTEGER DEFAULT 0,
    bluffs_caught INTEGER DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at  DATETIME
);

CREATE TABLE IF NOT EXISTS telemetry (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT    NOT NULL,
    session_id TEXT,
    payload    TEXT,
    ts         DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


async def init_db(db_path: str = DB_PATH) -> None:
    """
    Create all required tables and indexes.
    Safe to run multiple times.

    Args:
        db_path: Path to the SQLite database file.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()
    logger.info(f"Database initialized at {db_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_db())
