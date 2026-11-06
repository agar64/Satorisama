from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class LookupResult:
    surface: str
    reading: str | None
    gloss: str | None
    kind: str  # "word" | "name" | "kanji"
    entry_id: int | None = None


class DictDB:
    def __init__(self, db_path: str):
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row

    def close(self):
        self.con.close()

    def lookup_surface(self, surface: str) -> LookupResult | None:
        # Word lookup
        row = self.con.execute(
            "SELECT surface, reading, gloss, score FROM term WHERE surface=? ORDER BY score DESC LIMIT 1",
            (surface,),
        ).fetchone()
        if row:
            return LookupResult(surface=row["surface"], reading=row["reading"], gloss=row["gloss"], kind="word")

        # Name fallback
        row = self.con.execute(
            "SELECT surface, reading, gloss, score FROM name_term WHERE surface=? ORDER BY score DESC LIMIT 1",
            (surface,),
        ).fetchone()
        if row:
            return LookupResult(surface=row["surface"], reading=row["reading"], gloss=row["gloss"], kind="name")

        # Kanji fallback
        if len(surface) == 1:
            k = self.con.execute("SELECT * FROM kanji WHERE literal=?", (surface,)).fetchone()
            if k:
                gloss = k["meaning"]
                reading = " | ".join([x for x in [k["on_yomi"], k["kun_yomi"]] if x])
                return LookupResult(surface=surface, reading=reading, gloss=gloss, kind="kanji")
        return None

    def longest_match_from_many(self, text: str, start: int, max_len: int = 24, limit: int = 5) -> tuple[str, list[
        LookupResult]] | None:
        if start < 0: start = 0
        if start >= len(text): return None
        tail = text[start:]
        Lmax = min(len(tail), max_len)
        for L in range(Lmax, 0, -1):
            s = tail[:L]
            hits = list(self._lookup_surface_many_cached(s, limit))
            if hits:
                return s, hits
        return None

    def lookup_surface_many(self, surface: str, limit: int = 5) -> list[LookupResult]:
        out: list[LookupResult] = []

        for table, kind in (("term", "word"), ("name_term", "name")):
            rows = self.con.execute(
                f"SELECT surface, reading, gloss, score FROM {table} WHERE surface=? ORDER BY score DESC LIMIT ?",
                (surface, limit),
            ).fetchall()
            for r in rows:
                out.append(LookupResult(surface=r["surface"],
                                        reading=r["reading"],
                                        gloss=r["gloss"],
                                        kind=kind))
            if out:
                return out

        # kanji fallback (single)
        if len(surface) == 1:
            k = self.con.execute("SELECT * FROM kanji WHERE literal=?", (surface,)).fetchone()
            if k:
                gloss = k["meaning"]
                reading = " | ".join([x for x in [k["on_yomi"], k["kun_yomi"]] if x])
                return [LookupResult(surface=surface, reading=reading, gloss=gloss, kind="kanji")]

        return []

    @lru_cache(maxsize=20000)
    def _lookup_surface_many_cached(self, surface: str, limit: int) -> tuple[LookupResult, ...]:
        return tuple(self.lookup_surface_many(surface, limit=limit))
