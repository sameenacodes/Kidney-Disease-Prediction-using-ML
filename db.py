"""SQLite storage for assessments (absolute path, so it never depends on the working directory)."""
import json, os, sqlite3
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("CKD_DB", os.path.join(BASE, "data", "patient_history.db"))


def conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS assessments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age REAL, sex TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            values_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            prediction TEXT, risk TEXT, probability REAL, egfr REAL, stage TEXT,
            created_at TEXT NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_name ON assessments(patient_name COLLATE NOCASE)")


def _row(r):
    if r is None:
        return None
    d = dict(r)
    d["values"] = json.loads(d.pop("values_json"))
    d["result"] = json.loads(d.pop("result_json"))
    return d


def save(name, sex, source, values, result):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO assessments(patient_name,age,sex,source,values_json,result_json,prediction,risk,probability,egfr,stage,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, values.get("age"), sex, source, json.dumps(values), json.dumps(result), result["prediction"],
             result["risk"], result["probability"], result["egfr"], result["stage"],
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return cur.lastrowid


def get(aid):
    with conn() as c:
        return _row(c.execute("SELECT * FROM assessments WHERE id=?", (aid,)).fetchone())


def delete(aid):
    with conn() as c:
        c.execute("DELETE FROM assessments WHERE id=?", (aid,))


def search(q="", flt="all"):
    sql, args = "SELECT * FROM assessments WHERE 1=1", []
    if q:
        if q.strip().lstrip("#").isdigit():
            sql += " AND (id=? OR patient_name LIKE ?)"; args += [int(q.strip().lstrip("#")), f"%{q}%"]
        else:
            sql += " AND patient_name LIKE ?"; args.append(f"%{q}%")
    if flt == "ckd":
        sql += " AND prediction='CKD Detected'"
    elif flt == "nockd":
        sql += " AND prediction='No CKD Detected'"
    elif flt in ("High", "Moderate", "Low"):
        sql += " AND risk=?"; args.append(flt)
    sql += " ORDER BY id DESC"
    with conn() as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def by_patient(name):
    with conn() as c:
        rows = c.execute("SELECT * FROM assessments WHERE patient_name=? COLLATE NOCASE ORDER BY created_at, id", (name,)).fetchall()
        return [_row(r) for r in rows]


def stats():
    rows = search()
    s = {"total": len(rows), "ckd": 0, "nockd": 0, "high": 0, "moderate": 0, "low": 0, "patients": len({r["patient_name"].lower() for r in rows}),
         "stages": {}, "per_day": {}, "avg_prob": 0.0, "review": 0, "sources": {}}
    for r in rows:
        s["ckd" if r["prediction"] == "CKD Detected" else "nockd"] += 1
        s[r["risk"].lower()] += 1
        if r["stage"]:
            s["stages"][r["stage"]] = s["stages"].get(r["stage"], 0) + 1
        day = r["created_at"][:10]
        s["per_day"][day] = s["per_day"].get(day, 0) + 1
        s["sources"][r["source"]] = s["sources"].get(r["source"], 0) + 1
        s["review"] += 1 if r["result"].get("needs_review") else 0
    if rows:
        s["avg_prob"] = round(sum(r["probability"] for r in rows) / len(rows) * 100, 1)
    s["per_day"] = dict(sorted(s["per_day"].items())[-14:])
    return s
