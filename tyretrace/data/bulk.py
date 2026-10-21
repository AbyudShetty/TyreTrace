"""Bulk pull: every race weekend in a year range -> the data lake.

    python -m tyretrace.data.bulk                 # 2023 .. today via OpenF1, resumable
    python -m tyretrace.data.bulk --assemble      # just rebuild the lake files

Every session lands in its own parquet under out/lake/sessions/, and a
manifest records what happened to each one. An interrupted pull resumes where
it stopped and never refetches a session it already has. Failures are kept in
the manifest with their error -- a missing session is a fact to report, not a
gap to paper over.

Assembled outputs (same canonical schema as everything else):

    out/lake/race.parquet  out/lake/quali.parquet  out/lake/practice.parquet

Sprint weekends have one practice session and a sprint; the sprint is not
pulled -- its fuel load and tyre rules differ from a grand prix, so mixing it
into the race arm would add a confounder, not data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import polars as pl

from tyretrace import schema
from tyretrace.data import fastf1_pull as fp

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "out" / "lake"
SESS = LAKE / "sessions"
MANIFEST = LAKE / "manifest.parquet"

KINDS = {"FP1": "practice", "FP2": "practice", "FP3": "practice", "Q": "quali", "R": "race"}


def weekends(first: int, last: int, before: dt.datetime) -> list[tuple[int, int, str, list[str]]]:
    """(year, round, location, sessions-to-pull) for every event already run."""
    import fastf1

    out = []
    for y in range(first, last + 1):
        sched = fastf1.get_event_schedule(y, include_testing=False)
        for _, ev in sched.iterrows():
            if ev["EventDate"].to_pydatetime() >= before:
                continue
            names = [str(ev.get(f"Session{i}", "")) for i in range(1, 6)]
            ses = []
            for n in names:
                code = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
                        "Qualifying": "Q", "Race": "R"}.get(n)
                if code:
                    ses.append(code)
            out.append((y, int(ev["RoundNumber"]), str(ev["Location"]), ses))
    return out


def _manifest() -> pl.DataFrame:
    if MANIFEST.exists():
        return pl.read_parquet(MANIFEST)
    return pl.DataFrame(schema={"uid": pl.Utf8, "year": pl.Int16, "round": pl.Int8,
                                "session": pl.Utf8, "status": pl.Utf8, "laps": pl.Int32,
                                "error": pl.Utf8, "seconds": pl.Float64})


def _record(m: pl.DataFrame, row: dict) -> pl.DataFrame:
    m = m.filter(pl.col("uid") != row["uid"])
    m = pl.concat([m, pl.DataFrame([row], schema=m.schema)])
    tmp = MANIFEST.with_suffix(".tmp")
    m.write_parquet(tmp)
    tmp.replace(MANIFEST)
    return m


def pull_openf1(first: int, last: int, retry_failed: bool = False) -> pl.DataFrame:
    """The default source since F1's live-timing archive went behind a login
    (see tyretrace.data.openf1). Same manifest, same session files."""
    from tyretrace.data import openf1

    SESS.mkdir(parents=True, exist_ok=True)
    m = _manifest()
    done = set(m.filter(pl.col("status") == "ok")["uid"].to_list())
    failed = set(m.filter(pl.col("status") != "ok")["uid"].to_list())
    plan = [s for y in range(first, last + 1) for s in openf1.sessions(y, dt.datetime.now())]
    todo = [s for s in plan if s["uid"] not in done and (retry_failed or s["uid"] not in failed)]
    print(f"{len(plan)} sessions in range · {len(done)} already pulled · {len(todo)} to fetch",
          flush=True)
    t_start = time.time()
    for i, meta in enumerate(todo, 1):
        t0 = time.time()
        try:
            df = openf1.load_session(meta)
            err = None if df.height else "no laps returned"
        except Exception as exc:                      # noqa: BLE001
            df, err = schema.empty(), f"{type(exc).__name__}: {exc}"[:300]
        secs = time.time() - t0
        if df.height:
            df.write_parquet(SESS / f"{meta['uid']}.parquet")
        m = _record(m, {"uid": meta["uid"], "year": meta["year"], "round": meta["round"],
                        "session": meta["code"], "status": "ok" if df.height else "empty",
                        "laps": df.height, "error": err, "seconds": secs})
        rate = (time.time() - t_start) / i
        print(f"  [{i:>4}/{len(todo)}] {meta['uid']:<12s} {meta['location']:<16.16s} "
              f"{df.height:>5} laps  {secs:5.1f}s   eta {rate * (len(todo) - i) / 60:5.1f} min",
              flush=True)
    return m


def rebuild(first: int, last: int) -> int:
    """Re-derive every pulled OpenF1 session from cached responses -- after a
    change to the adapter, without a single new request for laps."""
    from tyretrace.data import openf1

    m = _manifest()
    ok = set(m.filter(pl.col("status") == "ok")["uid"].to_list())
    metas = [s for y in range(first, last + 1) for s in openf1.sessions(y, dt.datetime.now())
             if s["uid"] in ok]
    for i, meta in enumerate(metas, 1):
        df = openf1.load_session(meta)
        if df.height:
            df.write_parquet(SESS / f"{meta['uid']}.parquet")
        if i % 50 == 0:
            print(f"  rebuilt {i}/{len(metas)}", flush=True)
    return len(metas)


def pull(first: int, last: int, retry_failed: bool = False) -> pl.DataFrame:
    import fastf1

    fastf1.set_log_level("ERROR")
    fp.enable_cache()
    SESS.mkdir(parents=True, exist_ok=True)
    m = _manifest()
    done = set(m.filter(pl.col("status") == "ok")["uid"].to_list())
    failed = set(m.filter(pl.col("status") != "ok")["uid"].to_list())

    plan = [(y, r, loc, s) for y, r, loc, ss in weekends(first, last, dt.datetime.now())
            for s in ss]
    todo = [p for p in plan if f"{p[0]}_{p[1]:02d}_{p[3]}" not in done
            and (retry_failed or f"{p[0]}_{p[1]:02d}_{p[3]}" not in failed)]
    print(f"{len(plan)} sessions in range · {len(done)} already pulled · {len(todo)} to fetch",
          flush=True)

    t_start = time.time()
    for i, (y, r, loc, s) in enumerate(todo, 1):
        uid = f"{y}_{r:02d}_{s}"
        t0 = time.time()
        df = fp.load_session(y, r, s)
        secs = time.time() - t0
        if df.height:
            df.write_parquet(SESS / f"{uid}.parquet")
            m = _record(m, {"uid": uid, "year": y, "round": r, "session": s, "status": "ok",
                            "laps": df.height, "error": None, "seconds": secs})
        else:
            m = _record(m, {"uid": uid, "year": y, "round": r, "session": s, "status": "empty",
                            "laps": 0, "error": "no laps returned", "seconds": secs})
        rate = (time.time() - t_start) / i
        print(f"  [{i:>4}/{len(todo)}] {uid:<12s} {loc:<16.16s} {df.height:>5} laps  "
              f"{secs:5.1f}s   eta {rate * (len(todo) - i) / 60:5.1f} min", flush=True)
    return m


def assemble() -> dict[str, int]:
    """Concatenate session files into one canonical file per session kind."""
    files = sorted(SESS.glob("*.parquet"))
    out = {}
    for kind in ("practice", "quali", "race"):
        parts = [pl.read_parquet(f) for f in files if KINDS.get(f.stem.rsplit("_", 1)[1]) == kind]
        df = pl.concat(parts) if parts else schema.empty()
        df.write_parquet(LAKE / f"{kind}.parquet")
        out[kind] = df.height
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("openf1", "fastf1"), default="openf1",
                    help="fastf1 needs F1's live-timing archive, which now returns 401")
    ap.add_argument("--first", type=int, default=2023)
    ap.add_argument("--last", type=int, default=dt.date.today().year)
    ap.add_argument("--assemble", action="store_true", help="only rebuild lake files")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-derive pulled sessions from cached OpenF1 responses")
    a = ap.parse_args(argv)
    if a.rebuild:
        print(f"rebuilt {rebuild(a.first, a.last)} sessions from cache")
    elif not a.assemble:
        m = (pull_openf1 if a.source == "openf1" else pull)(a.first, a.last, a.retry_failed)
        bad = m.filter(pl.col("status") != "ok")
        print(f"\nmanifest: {m.filter(pl.col('status') == 'ok').height} ok · {bad.height} not ok")
    counts = assemble()
    print("lake:", ", ".join(f"{k} {v:,} laps" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
