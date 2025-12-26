PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS audio (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256        TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  data          BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS model (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS embedding (
  audio_id      INTEGER NOT NULL,
  model_id      INTEGER NOT NULL,
  vector_f32    BLOB NOT NULL,

  PRIMARY KEY (audio_id, model_id),

  FOREIGN KEY(audio_id) REFERENCES audio(id) ON DELETE CASCADE,
  FOREIGN KEY(model_id) REFERENCES model(id) ON DELETE CASCADE
);
