import hashlib
import io
import sqlite3
from typing import List, Optional, Tuple

from data.const import DB_PATH


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


class DbManager:
    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None
        if DB_PATH.exists():
            try:
                self.conn = sqlite3.connect(
                    DB_PATH, check_same_thread=False, isolation_level=None
                )
            except sqlite3.Error:
                self.conn = None
        self.is_connected = self.conn is not None

    def insert_audio_if_not_exists(self, name: str, data: bytes) -> bool:
        """Inserts audio if it doesn't exist. Returns True if inserted."""
        if not self.is_connected:
            return False
        digest = _sha256_bytes(data)
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM audio WHERE sha256 = ?", (digest,))
            if cursor.fetchone():
                return False  # Already exists
            cursor.execute(
                "INSERT INTO audio (sha256, original_name, data) VALUES (?, ?, ?)",
                (digest, name, data),
            )
            return cursor.rowcount == 1

    def get_audio_files(self) -> List[Tuple[int, str]]:
        """Returns a list of (id, original_name) for all audio files."""
        if not self.is_connected:
            return []
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, original_name FROM audio ORDER BY original_name")
            return cursor.fetchall()

    def get_audio_data(self, audio_id: int) -> Optional[Tuple[io.BytesIO, str]]:
        """Returns (audio_data, original_name) for a given audio_id."""
        if not self.is_connected:
            return None, None
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT data, original_name FROM audio WHERE id = ?",
                (audio_id,)
                )
            row = cursor.fetchone()
            if row:
                return io.BytesIO(row[0]), row[1]
        return None, None

    def __del__(self):
        if self.conn:
            self.conn.close()
