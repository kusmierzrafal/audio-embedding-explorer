import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "app.db"


def scalar(conn, q):
    return conn.execute(q).fetchone()[0]


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    print("audio     :", scalar(conn, "SELECT COUNT(*) FROM audio;"))
    print("model     :", scalar(conn, "SELECT COUNT(*) FROM model;"))
    print("embedding :", scalar(conn, "SELECT COUNT(*) FROM embedding;"))

    print("\nModels:")
    for mid, name in conn.execute("SELECT id, name FROM model ORDER BY id;"):
        print(f"  {mid}  {name}")

    conn.close()


if __name__ == "__main__":
    main()
