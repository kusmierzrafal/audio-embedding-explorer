import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")

        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """).fetchall()

        print("Tables:")
        for row in tables:
            print(f" - {row['name']}")

        print("\nForeign key check:")
        fk_issues = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if not fk_issues:
            print(" ✅ OK (no issues)")
        else:
            print(" ❌ Issues found:")
            for issue in fk_issues:
                print(dict(issue))


if __name__ == "__main__":
    main()
