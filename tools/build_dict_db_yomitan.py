from __future__ import annotations
import argparse, json, sqlite3, zipfile
from pathlib import Path
from typing import Any

SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS term (
  surface TEXT NOT NULL,
  reading TEXT,
  gloss TEXT,
  dict_title TEXT,
  score REAL DEFAULT 0,
  PRIMARY KEY(surface, reading, gloss, dict_title)
);
CREATE INDEX IF NOT EXISTS idx_term_surface ON term(surface);

CREATE TABLE IF NOT EXISTS kanji (
  literal TEXT PRIMARY KEY,
  on_yomi TEXT,
  kun_yomi TEXT,
  meaning TEXT,
  dict_title TEXT
);
CREATE TABLE IF NOT EXISTS name_term (
  surface TEXT NOT NULL,
  reading TEXT,
  gloss TEXT,
  dict_title TEXT,
  score REAL DEFAULT 0,
  PRIMARY KEY(surface, reading, gloss, dict_title)
);
CREATE INDEX IF NOT EXISTS idx_name_term_surface ON name_term(surface);
"""

def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)
    return con

def read_json_from_zip(z: zipfile.ZipFile, name: str) -> Any:
    with z.open(name) as f:
        return json.load(f)

def get_title(z: zipfile.ZipFile) -> str:
    try:
        idx = read_json_from_zip(z, "index.json")
        # common: {"title": "..."} or {"title": {"en": "..."}}
        t = idx.get("title", "unknown")
        if isinstance(t, dict):
            return t.get("en") or next(iter(t.values()), "unknown")
        return str(t)
    except Exception:
        return "unknown"

def iter_bank_names(z: zipfile.ZipFile, prefix: str) -> list[str]:
    names = [n for n in z.namelist() if n.startswith(prefix) and n.endswith(".json")]
    # sort by trailing number if present
    def key(n: str):
        stem = Path(n).stem  # term_bank_12
        try:
            return int(stem.split("_")[-1])
        except:
            return 10**9
    return sorted(names, key=key)

def _collect_text(x: Any, out: list[str]) -> None:
    if x is None:
        return
    if isinstance(x, str):
        s = x.strip()
        if s:
            out.append(s)
        return
    if isinstance(x, list):
        for item in x:
            _collect_text(item, out)
        return
    if isinstance(x, dict):
        # common keys seen in yomitan structured gloss
        for k in ("text", "content", "value", "gloss"):
            if k in x:
                _collect_text(x[k], out)
        # sometimes nested arrays under "items"
        if "items" in x:
            _collect_text(x["items"], out)
        return

def extract_gloss_string(gloss_field: Any) -> str:
    parts: list[str] = []
    _collect_text(gloss_field, parts)
    # keep it short-ish for popup
    parts = [p for p in parts if p]
    return "; ".join(parts[:6])


def parse_term_entry(row: Any) -> tuple[str, str | None, str, float]:
    """
    Defensive parse for Yomitan term rows (array-based).
    Common layout (varies by rev):
      [term, reading, defTags, rules, score, glossary, ...]
    We will:
      - surface: row[0]
      - reading: row[1] if string else None
      - score: row[4] if number else 0
      - gloss: row[5] if present
    """
    if not isinstance(row, list) or len(row) < 2:
        raise ValueError("Bad term row")
    surface = str(row[0])

    reading = row[1] if isinstance(row[1], str) and row[1].strip() else None

    score = 0.0
    if len(row) > 4 and isinstance(row[4], (int, float)):
        score = float(row[4])

    gloss = ""
    if len(row) > 5:
        gloss = extract_gloss_string(row[5])

    return surface, reading, gloss, score

def parse_kanji_entry(row: Any) -> tuple[str, str, str, str]:
    """
    Defensive parse for Yomitan kanji rows.
    Common layouts vary a lot; usually:
      [literal, onyomi, kunyomi, tags, meanings, ...]
    We'll try:
      literal=row[0]
      on=row[1] if str
      kun=row[2] if str
      meaning=row[4] if list/str
    """
    if not isinstance(row, list) or not row:
        raise ValueError("Bad kanji row")
    literal = str(row[0])

    on = row[1] if len(row) > 1 and isinstance(row[1], str) else ""
    kun = row[2] if len(row) > 2 and isinstance(row[2], str) else ""

    meaning = ""
    if len(row) > 4:
        meaning = extract_gloss_string(row[4])
    elif len(row) > 3:
        meaning = extract_gloss_string(row[3])

    return literal, on, kun, meaning

def ingest_yomitan_zip(zip_path: Path, con: sqlite3.Connection, target: str = "term") -> None:
    assert target in ("term", "name_term")
    with zipfile.ZipFile(zip_path, "r") as z:
        title = get_title(z)
        print(f"Reading {zip_path.name}  (title={title})", flush=True)

        cur = con.cursor()

        # Terms
        term_files = iter_bank_names(z, "term_bank_")
        tcount = 0
        for name in term_files:
            data = read_json_from_zip(z, name)
            if not isinstance(data, list):
                continue
            for row in data:
                try:
                    surface, reading, gloss, score = parse_term_entry(row)
                except Exception:
                    continue
                if not surface:
                    continue
                cur.execute(
                    f"INSERT OR IGNORE INTO {target}(surface, reading, gloss, dict_title, score) VALUES (?,?,?,?,?)",
                    (surface, reading, gloss, title, score),
                )
                tcount += 1
            if tcount and tcount % 50000 == 0:
                con.commit()
                print("  terms:", tcount, flush=True)

        # Kanji (if present)
        k_files = iter_bank_names(z, "kanji_bank_")
        kcount = 0
        for name in k_files:
            data = read_json_from_zip(z, name)
            if not isinstance(data, list):
                continue
            for row in data:
                try:
                    literal, on, kun, meaning = parse_kanji_entry(row)
                except Exception:
                    continue
                if not literal:
                    continue
                cur.execute(
                    "INSERT OR REPLACE INTO kanji(literal, on_yomi, kun_yomi, meaning, dict_title) VALUES (?,?,?,?,?)",
                    (literal, on, kun, meaning, title),
                )
                kcount += 1
            if kcount and kcount % 5000 == 0:
                con.commit()
                print("  kanji:", kcount, flush=True)

        con.commit()
        print(f"Done: terms={tcount} kanji={kcount}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="append", required=True, help="Path to Yomitan-format dictionary ZIP (repeatable)")
    ap.add_argument("--out", default="data/jp_dict.sqlite")
    ap.add_argument("--names-zip", action="append", default=[], help="Yomitan-format JMnedict ZIP(s)")
    args = ap.parse_args()

    out = Path(args.out)
    con = open_db(out)

    # wipe tables (fresh build)
    con.execute("DELETE FROM term")
    con.execute("DELETE FROM kanji")
    con.execute("DELETE FROM name_term")
    con.commit()

    for zp in args.zip:
        ingest_yomitan_zip(Path(zp), con)

    for zp in args.zip:
        ingest_yomitan_zip(Path(zp), con, target="term")

    for zp in args.names_zip:
        ingest_yomitan_zip(Path(zp), con, target="name_term")

    con.execute("CREATE INDEX IF NOT EXISTS idx_term_surface ON term(surface)")
    con.commit()
    con.close()
    print("Wrote:", out)

if __name__ == "__main__":
    main()
