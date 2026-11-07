import hashlib
import json
import sqlite3
from pathlib import Path


def make_key(fields):
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class PassCache:
    """Stores every scoring pass on disk, keyed by a hash of its full configuration."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS passes (key TEXT PRIMARY KEY, meta TEXT, value TEXT)"
        )
        self.conn.commit()

    def get(self, key):
        row = self.conn.execute("SELECT value FROM passes WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key, meta, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO passes VALUES (?, ?, ?)",
            (key, json.dumps(meta, sort_keys=True), json.dumps(value)),
        )
        self.conn.commit()

    def __len__(self):
        return self.conn.execute("SELECT COUNT(*) FROM passes").fetchone()[0]

    def close(self):
        self.conn.close()
