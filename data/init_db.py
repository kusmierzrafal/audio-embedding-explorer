import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema file not found: {SCHEMA_PATH}")
        print("Make sure it exists at: data/schema.sql")
        return 1

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Enable foreign key constraints in SQLite
            conn.execute("PRAGMA foreign_keys = ON;")
            # Execute all SQL statements from the schema file
            conn.executescript(schema_sql)

        print(f"✅ Database initialized successfully: {DB_PATH}")
        return 0

    except sqlite3.Error as exc:
        print("❌ SQLite error:", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
