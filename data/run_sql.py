import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "app.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sql_file")
    args = ap.parse_args()

    sql_path = Path(args.sql_file).resolve()
    sql = sql_path.read_text(encoding="utf-8")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(sql)
    conn.commit()
    conn.close()

    print(f"✅ Executed: {sql_path}")


if __name__ == "__main__":
    main()
