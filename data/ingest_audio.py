import argparse
import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "app.db"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def iter_files(root: Path, recursive: bool):
    if root.is_file():
        yield root
        return
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="File or directory with audio files")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    inserted = 0
    skipped = 0

    for f in iter_files(root, args.recursive):
        data = f.read_bytes()
        digest = sha256_bytes(data)

        cur = conn.execute(
            "INSERT OR IGNORE INTO audio (sha256, original_name, data) "
            "VALUES (?, ?, ?);",
            (digest, f.name, data),
        )
        if cur.rowcount == 1:
            inserted += 1
            print(f"✅ Inserted: {f.name} ({digest[:12]}...)")
        else:
            skipped += 1
            print(f"↩️ Skipped:  {f.name} ({digest[:12]}...)")

    conn.commit()
    conn.close()

    print(f"\nDone. Inserted={inserted}, Skipped={skipped}")


if __name__ == "__main__":
    main()
