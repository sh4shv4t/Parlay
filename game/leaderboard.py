"""SQLite-backed leaderboard for Parlay."""
import logging
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "parlay.db"


class Leaderboard:
    """Async SQLite-backed leaderboard."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path

    async def record_result(
        self,
        player_name: str,
        scenario_id: str,
        persona: str,
        total_reward: float,
        deal_efficiency: float,
        acts_completed: int,
        deal_closed: bool,
    ) -> int:
        """Insert a game result. Returns the new row id."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO leaderboard
                    (player_name, scenario_id, persona, total_reward,
                     deal_efficiency, acts_completed, deal_closed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    player_name, scenario_id, persona, total_reward,
                    deal_efficiency, acts_completed, int(deal_closed),
                ),
            )
            await db.commit()
            logger.info(
                f"Leaderboard: recorded result for {player_name!r}, "
                f"reward={total_reward:.2f}"
            )
            return cursor.lastrowid

    async def get_top(
        self,
        scenario_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return top entries, optionally filtered by scenario."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if scenario_id:
                cursor = await db.execute(
                    """
                    SELECT player_name, scenario_id, persona, total_reward,
                           deal_efficiency, acts_completed, deal_closed, created_at
                    FROM leaderboard
                    WHERE scenario_id = ?
                    ORDER BY total_reward DESC
                    LIMIT ?
                    """,
                    (scenario_id, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT player_name, scenario_id, persona, total_reward,
                           deal_efficiency, acts_completed, deal_closed, created_at
                    FROM leaderboard
                    ORDER BY total_reward DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_rank(
        self,
        player_name: str,
        scenario_id: Optional[str] = None,
    ) -> int:
        """Return the player's best rank (1-indexed). Returns 0 if not found."""
        async with aiosqlite.connect(self.db_path) as db:
            if scenario_id:
                cursor = await db.execute(
                    """
                    SELECT COUNT(*) + 1 FROM leaderboard
                    WHERE total_reward > (
                        SELECT MAX(total_reward) FROM leaderboard
                        WHERE player_name = ? AND scenario_id = ?
                    ) AND scenario_id = ?
                    """,
                    (player_name, scenario_id, scenario_id),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT COUNT(*) + 1 FROM leaderboard
                    WHERE total_reward > (
                        SELECT MAX(total_reward) FROM leaderboard
                        WHERE player_name = ?
                    )
                    """,
                    (player_name,),
                )
            row = await cursor.fetchone()
            return row[0] if row and row[0] else 0
