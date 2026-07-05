"""
diagnostics.py — `python main.py --doctor` health check.

Answers "why is nothing happening / what's misconfigured?" in one command.
Runs a series of independent probes and prints a pass/warn/fail table:

  • LLM providers   — configured? reachable? (a real 1-token ping)
  • Yahoo Finance   — can we fetch prices?
  • SEC EDGAR       — reachable for point-in-time fundamentals?
  • lxml            — installed? (needed for earnings-surprise features)
  • ML models       — present and fresh per watchlist ticker?
  • Feedback CSVs   — present and schema-intact (73 columns)?

Every probe is isolated: a failure in one never aborts the rest. Exit code
is 0 when nothing FAILED (warnings are allowed), 1 when any probe FAILED —
so it doubles as a CI/pre-flight smoke test.
"""

import glob
import os
import datetime

from config import (
    LLM_ENABLED, AZURE_ENABLED, GOOGLE_ENABLED, DEEPSEEK_ENABLED,
    ML_MODELS_DIR, ML_RETRAIN_DAYS, FEEDBACK_DIR,
)

# Status constants
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_STYLE = {PASS: "green", WARN: "yellow", FAIL: "red"}
_ICON  = {PASS: "✓", WARN: "!", FAIL: "✗"}


def _tickers_for_check() -> list[str]:
    """Watchlist tickers to check models/feedback for (tickers.txt, capped)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.txt")
    out: list[str] = []
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    tok = line.split("#", 1)[0].strip().upper()
                    if tok:
                        out.append(tok)
        except OSError:
            pass
    return out[:10]


# ------------------------------------------------------------------ #
# Individual probes — each returns a list of (name, status, detail)
# ------------------------------------------------------------------ #

def _check_llm() -> list[tuple]:
    rows: list[tuple] = []
    if not LLM_ENABLED:
        rows.append(("LLM providers", WARN,
                     "none configured — runs use the rule-based judge (--no-llm equivalent)"))
        return rows
    configured = [n for n, on in (("Azure", AZURE_ENABLED),
                                  ("Google", GOOGLE_ENABLED),
                                  ("DeepSeek", DEEPSEEK_ENABLED)) if on]
    rows.append(("LLM configured", PASS, ", ".join(configured)))
    try:
        from analysis.llm_analysis import ping_providers
        for r in ping_providers():
            if r["ok"]:
                rows.append((f"  ping {r['provider']}", PASS, f"reachable ({r['latency_s']}s)"))
            else:
                status = FAIL if r["connection_error"] else WARN
                hint = " — unreachable (network/endpoint/key)" if r["connection_error"] else ""
                rows.append((f"  ping {r['provider']}", status, r["detail"] + hint))
    except Exception as e:
        rows.append(("  ping providers", FAIL, f"ping harness error: {e}"))
    return rows


def _check_yahoo() -> list[tuple]:
    try:
        from data import market_data
        df = market_data.get_history("AAPL", "5d")
        if df is not None and not df.empty:
            return [("Yahoo Finance", PASS, f"prices OK (AAPL last close ${float(df['Close'].iloc[-1]):.2f})")]
        return [("Yahoo Finance", FAIL, "empty response for AAPL — Yahoo may be rate-limiting")]
    except Exception as e:
        return [("Yahoo Finance", FAIL, f"{type(e).__name__}: {str(e)[:120]}")]


def _check_edgar() -> list[tuple]:
    try:
        from data import edgar_data
        cik = edgar_data.get_cik("AAPL")
        if cik:
            return [("SEC EDGAR", PASS, f"reachable (AAPL CIK {cik}) — fundamentals available")]
        return [("SEC EDGAR", WARN, "no CIK for AAPL — EDGAR fundamentals will be NaN")]
    except Exception as e:
        return [("SEC EDGAR", WARN, f"unreachable ({type(e).__name__}) — fundamentals degrade to NaN")]


def _check_lxml() -> list[tuple]:
    try:
        import lxml  # noqa: F401
        return [("lxml (earnings surprise)", PASS, "installed — earnings-surprise features active")]
    except Exception:
        return [("lxml (earnings surprise)", WARN,
                 "missing — earnings-surprise features are NaN; `pip install lxml`")]


def _check_models() -> list[tuple]:
    tickers = _tickers_for_check()
    if not tickers:
        return [("ML models", WARN, "no tickers.txt — nothing to check")]
    cutoff = datetime.datetime.now() - datetime.timedelta(days=ML_RETRAIN_DAYS)
    fresh, stale, missing = [], [], []
    for t in tickers:
        clf = os.path.join(ML_MODELS_DIR, t, "clf_5d.pkl")
        if not os.path.isfile(clf):
            missing.append(t)
        elif datetime.datetime.fromtimestamp(os.path.getmtime(clf)) < cutoff:
            stale.append(t)
        else:
            fresh.append(t)
    rows = [("ML models fresh", PASS if not stale and not missing else WARN,
             f"{len(fresh)} fresh, {len(stale)} stale (>{ML_RETRAIN_DAYS}d), {len(missing)} untrained")]
    if stale:
        rows.append(("  stale — will auto-retrain", WARN, ", ".join(stale)))
    if missing:
        rows.append(("  untrained — train on next run", WARN, ", ".join(missing)))
    # Global pooled model
    if os.path.isfile(os.path.join(ML_MODELS_DIR, "_GLOBAL", "clf_5d.pkl")):
        rows.append(("Global pooled model", PASS, "present (blended into predictions)"))
    else:
        rows.append(("Global pooled model", WARN, "absent — run `python main.py --retrain-global`"))
    return rows


def _check_feedback() -> list[tuple]:
    from feedback.tracker import COLUMNS
    expected = len(COLUMNS)
    files = glob.glob(os.path.join(FEEDBACK_DIR, "*_feedback.csv"))
    if not files:
        return [("Feedback CSVs", WARN, "none yet — the learning loop starts after your first run")]
    ok, bad = 0, []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip()
            ncols = header.count(",") + 1
            if ncols == expected:
                ok += 1
            else:
                bad.append(f"{os.path.basename(path)} ({ncols} cols)")
        except Exception:
            bad.append(f"{os.path.basename(path)} (unreadable)")
    status = PASS if not bad else FAIL
    detail = f"{ok} intact (73-col schema)"
    if bad:
        detail += f"; SCHEMA MISMATCH: {', '.join(bad[:5])}"
    return [("Feedback CSVs", status, detail)]


# ------------------------------------------------------------------ #
# Entry point (called from main.py --doctor)
# ------------------------------------------------------------------ #

def run_doctor(console) -> int:
    """Run all probes, print a table, return an exit code (0 ok / 1 failures)."""
    from rich.table import Table

    console.print("\n[bold cyan]FinBot — Doctor[/bold cyan]  (environment & connectivity check)\n")

    probes = [
        ("Connectivity", _check_llm),
        ("Connectivity", _check_yahoo),
        ("Connectivity", _check_edgar),
        ("Dependencies", _check_lxml),
        ("Local state",  _check_models),
        ("Local state",  _check_feedback),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", no_wrap=True)
    table.add_column("", justify="center", width=3)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")

    counts = {PASS: 0, WARN: 0, FAIL: 0}
    with console.status("[dim]Running probes…[/dim]"):
        for _group, fn in probes:
            try:
                rows = fn()
            except Exception as e:
                rows = [(fn.__name__, FAIL, f"probe crashed: {e}")]
            for name, status, detail in rows:
                counts[status] = counts.get(status, 0) + 1
                st = _STYLE[status]
                table.add_row(name, f"[{st}]{_ICON[status]}[/{st}]",
                              f"[{st}]{status}[/{st}]", detail)

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]{counts[PASS]} pass[/green] · "
        f"[yellow]{counts[WARN]} warn[/yellow] · "
        f"[red]{counts[FAIL]} fail[/red]"
    )
    if counts[FAIL]:
        console.print("[red]Some checks failed — see FAIL rows above.[/red]")
    elif counts[WARN]:
        console.print("[yellow]All critical checks passed; warnings are non-fatal.[/yellow]")
    else:
        console.print("[green]All systems go.[/green]")
    console.print()
    return 1 if counts[FAIL] else 0
