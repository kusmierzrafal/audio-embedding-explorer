import hashlib
import io
import sqlite3
from typing import List, Optional, Tuple

import numpy as np

from data.const import DB_PATH
from src.utils.audio_utils import AudioHelper, safe_tensor_to_numpy


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
                (audio_id,),
            )
            row = cursor.fetchone()
            if row:
                return io.BytesIO(row[0]), row[1]
        return None, None

    def get_or_insert_model(self, model_name: str) -> Optional[int]:
        """Gets model ID or creates new model entry. Returns model_id."""
        if not self.is_connected:
            return None
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM model WHERE name = ?", (model_name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            cursor.execute("INSERT INTO model (name) VALUES (?)", (model_name,))
            return cursor.lastrowid

    def get_audio_id_by_data(self, data: bytes) -> Optional[int]:
        """Gets audio ID by data hash."""
        if not self.is_connected:
            return None
        digest = _sha256_bytes(data)
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM audio WHERE sha256 = ?", (digest,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_embedding(self, audio_id: int, model_id: int) -> Optional[np.ndarray]:
        """Gets embedding vector from database."""
        if not self.is_connected:
            return None
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT vector_f32 FROM embedding WHERE audio_id = ? AND model_id = ?",
                (audio_id, model_id),
            )
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None

    def get_or_compute_audio_embedding(
        self, embedder, audio_data, audio_name: str, sr: int, model_name: str
    ):
        """Gets embedding from cache or computes and caches it."""

        if not self.is_connected:
            # No database, just compute
            if isinstance(audio_data, bytes):
                bio = io.BytesIO(audio_data)
                y, _ = AudioHelper.load_audio(bio, sr)
                emb = embedder.embed_audio(y, sr)
            else:
                emb = embedder.embed_audio(audio_data, sr)
            return getattr(emb, "vector", emb)

        # Get or create model ID
        model_id = self.get_or_insert_model(model_name)
        if not model_id:
            if isinstance(audio_data, bytes):
                bio = io.BytesIO(audio_data)
                y, _ = AudioHelper.load_audio(bio, sr)
                emb = embedder.embed_audio(y, sr)
            else:
                emb = embedder.embed_audio(audio_data, sr)
            return getattr(emb, "vector", emb)

        # Handle different input types
        if isinstance(audio_data, bytes):
            data_bytes = audio_data
            # Load audio for computation
            bio = io.BytesIO(audio_data)
            y, _ = AudioHelper.load_audio(bio, sr)
        else:
            # audio_data is already numpy array, compute without caching
            emb = embedder.embed_audio(audio_data, sr)
            return getattr(emb, "vector", emb)

        audio_id = self.get_audio_id_by_data(data_bytes)
        if not audio_id:
            self.insert_audio_if_not_exists(audio_name, data_bytes)
            audio_id = self.get_audio_id_by_data(data_bytes)

        if audio_id:
            # Try to get cached embedding
            cached_vector = self.get_embedding(audio_id, model_id)
            if cached_vector is not None:
                return cached_vector

        # Compute new embedding
        emb = embedder.embed_audio(y, sr)
        vector = getattr(emb, "vector", emb)

        # Save to cache
        if audio_id:
            # Convert tensor to numpy array, handling both CPU and CUDA tensors
            vector_np = safe_tensor_to_numpy(vector)
            self.save_embedding(audio_id, model_id, vector_np)

        return vector

    def save_embedding(self, audio_id: int, model_id: int, vector: np.ndarray) -> bool:
        """Saves embedding vector to database."""
        if not self.is_connected:
            return False
        vector_bytes = vector.astype(np.float32).tobytes()
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO embedding "
                "(audio_id, model_id, vector_f32) VALUES (?, ?, ?)",
                (audio_id, model_id, vector_bytes),
            )
            return cursor.rowcount == 1

    def __del__(self):
        if self.conn:
            self.conn.close()
