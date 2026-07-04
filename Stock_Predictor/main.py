"""
main.py — FinBot CLI entry point.

Usage:
    python main.py --ticker AAPL
    python main.py --ticker MSFT --no-llm

Orchestrates the full pipeline:
    1. Fetch stock price history + fundamentals     (data/stock_fetcher.py)
    2. Fetch analyst data                           (data/analyst_fetcher.py)
    3. Compute technical indicators                 (analysis/technical.py)
    4. Compute fundamental metrics                  (analysis/fundamental.py)
    5. Compute statistical models + Monte Carlo     (analysis/statistical.py)
    5b.Export CSV, score sentiment & run ML          (data/csv_exporter.py, ml/)
    6. Query LLM provider(s) for recommendation    (analysis/llm_analysis.py)
    7. Display full report in terminal              (output/terminal_display.py)
    8. Save Markdown report + PNG charts            (output/report_generator.py)
    9. Generate PDF report                           (output/pdf_generator.py)
"""

import argparse
import os
import sys
import time

# ------------------------------------------------------------------ #
# Force UTF-8 stdout/stderr so unicode symbols in the report (σ, β, …)
# never crash on Windows, where the console / a piped stream defaults
# to cp1252 and raises UnicodeEncodeError. Guarded for older Pythons
# and streams that don't expose reconfigure().
# ------------------------------------------------------------------ #
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# ---- Internal modules ---- #
from data.stock_fetcher    import fetch_stock_data, fetch_options_data
from data.analyst_fetcher  import fetch_analyst_data
from analysis.technical    import compute_all_technicals
from analysis.fundamental  import compute_all_fundamentals
from analysis.statistical  import compute_all_statistics
from analysis.llm_analysis    import get_llm_analysis, build_company_description
from analysis.rule_based_judge import run_rule_based_judge, run_rule_based_analysis
from output.terminal_display import display_full_report
from output.report_generator import create_report_dir, generate_markdown_report
from output.pdf_generator    import generate_pdf_report
from config import LLM_ENABLED, AZURE_ENABLED, GOOGLE_ENABLED, DEEPSEEK_ENABLED, GOOGLE_MODEL, DEEPSEEK_MODEL
from data.csv_exporter      import export_features_csv
from data.sentiment_fetcher import score_headlines
from ml.trainer    import load_or_train
from ml.predictor  import predict as ml_predict
from feedback.resolver import resolve_pending
from feedback.accuracy import get_context as get_accuracy_context, build_calibration_table
from feedback.tracker  import save_prediction as save_feedback, save_sentiment_history
from feedback.backfiller import backfill_historical_predictions, needs_backfill

console = Console()


# ------------------------------------------------------------------ #
# CLI argument parsing
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "finbot",
        description = "FinBot — AI-powered quantitative stock analyser",
        formatter_class = argparse.RawTextHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Input source — at most one of --ticker or --file may be supplied.
    # Standalone modes (--resolve, --backfill, --retrain-global,
    # --discover) may run without either, so the group itself is not
    # required; the check after parsing enforces that an analysis run
    # names its input.
    # ------------------------------------------------------------------
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--ticker", "-t",
        type = str,
        help = "Yahoo Finance ticker symbol (e.g. AAPL, MSFT, TSLA, 9988.HK)",
    )
    source.add_argument(
        "--file", "-f",
        type = str,
        metavar = "FILE",
        help = (
            "Path to a plain-text watchlist file with one ticker per line.\n"
            "Lines beginning with '#' and blank lines are ignored.\n"
            "Example: python main.py --file tickers.txt --no-llm"
        ),
    )

    parser.add_argument(
        "--no-llm",
        action  = "store_true",
        default = False,
        help    = "Skip LLM analysis entirely (useful for offline runs or testing)",
    )
    parser.add_argument(
        "--retrain",
        action  = "store_true",
        default = False,
        help    = "Force retraining of ML models even if cached models are fresh",
    )
    parser.add_argument(
        "--retrain-global",
        action  = "store_true",
        default = False,
        help    = (
            "(Re)train the pooled cross-ticker model (models/_GLOBAL/) from every\n"
            "ticker's most recent features.csv, then exit. Once trained, the\n"
            "predictor automatically blends it with per-ticker models and uses it\n"
            "alone for tickers with too little history to train on."
        ),
    )
    parser.add_argument(
        "--resolve",
        action  = "store_true",
        default = False,
        help    = (
            "Resolve pending predictions in feedback CSVs (fetch actual prices) "
            "and exit without running a full analysis."
        ),
    )
    parser.add_argument(
        "--backfill",
        action  = "store_true",
        default = False,
        help    = (
            "Backfill historical predictions from features.csv test split "
            "into the feedback CSV to bootstrap accuracy statistics, then exit."
        ),
    )
    parser.add_argument(
        "--discover",
        action  = "store_true",
        default = False,
        help    = (
            "Scan the market for new candidate tickers (yfinance screeners,\n"
            "volume spikes, 52-week-high breakouts, new listings), rank them\n"
            "with a quick rule-based score, and append the best to watchlist.txt."
        ),
    )
    parser.add_argument(
        "--discover-top",
        type    = int,
        default = 5,
        metavar = "N",
        help    = "How many top-ranked discoveries to append to watchlist.txt (default 5)",
    )
    parser.add_argument(
        "--discover-min-cap",
        type    = float,
        default = 100e6,
        metavar = "USD",
        help    = "Minimum market cap filter for discovered tickers (default 100000000)",
    )
    parser.add_argument(
        "--backtest",
        action  = "store_true",
        default = False,
        help    = (
            "Replay the rule-based judge's TECHNICAL lens weekly across each\n"
            "ticker's history and measure whether scores predicted 21d/63d\n"
            "forward returns (bucket table + rank IC). Uses --ticker/--file if\n"
            "given, else tickers.txt + watchlist.txt. Add --backtest-broad to\n"
            "pool ~110 large caps across all sectors for statistical power."
        ),
    )
    parser.add_argument(
        "--backtest-broad",
        action  = "store_true",
        default = False,
        help    = "Extend --backtest universe with the top ~10 US names per sector",
    )

    args = parser.parse_args()

    # Analysis runs need an input source; standalone maintenance modes
    # (--resolve/--backfill default to tickers.txt, --retrain-global and
    # --discover need none) do not.
    if not any((args.ticker, args.file, args.resolve, args.backfill,
                args.retrain_global, args.discover, args.backtest)):
        parser.error("one of --ticker/--file (or a standalone mode: "
                     "--resolve, --backfill, --retrain-global, --discover, "
                     "--backtest) is required")
    return args


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

def run_pipeline(ticker: str, skip_llm: bool = False, retrain: bool = False) -> None:
    """
    Execute the full FinBot analysis pipeline for *ticker*.

    Parameters
    ----------
    ticker   : str   — Yahoo Finance ticker symbol
    skip_llm : bool  — If True, bypass the LLM step (returns placeholder)
    retrain  : bool  — If True, force ML model retraining
    """

    ticker = ticker.upper().strip()

    # Initialise variables that are referenced across steps so they are always bound
    ml_result = None
    report_path = None
    pdf_path = None

    # ── Feedback: resolve pending predictions before new analysis ── #
    try:
        resolved_count = resolve_pending(ticker)
        if resolved_count > 0:
            console.log(f"[dim]Feedback: resolved {resolved_count} pending prediction(s) for {ticker}.[/dim]")
    except Exception as fb_err:
        console.log(f"[dim yellow]Feedback resolver error (non-fatal): {fb_err}[/dim yellow]")

    # ── Feedback: load historical accuracy context (per provider) ─── #
    mode_str = "no_llm" if skip_llm or not LLM_ENABLED else "llm"
    accuracy_context_map: dict = {}
    try:
        if mode_str == "llm":
            if AZURE_ENABLED:
                accuracy_context_map["Azure OpenAI"] = get_accuracy_context(
                    ticker, "llm", provider="Azure OpenAI"
                )
            if GOOGLE_ENABLED:
                accuracy_context_map[f"Google Gemini ({GOOGLE_MODEL})"] = get_accuracy_context(
                    ticker, "llm", provider=f"Google Gemini ({GOOGLE_MODEL})"
                )
            if DEEPSEEK_ENABLED:
                accuracy_context_map[f"DeepSeek ({DEEPSEEK_MODEL})"] = get_accuracy_context(
                    ticker, "llm", provider=f"DeepSeek ({DEEPSEEK_MODEL})"
                )
        else:
            accuracy_context_map["rule_based"] = get_accuracy_context(
                ticker, "no_llm", provider="rule_based"
            )
    except Exception:
        pass
    # Primary display context = first available provider's stats (or empty)
    accuracy_context = next(iter(accuracy_context_map.values()), {"has_data": False})

    steps = [
        "Fetching price history & fundamentals",
        "Fetching analyst data",
        "Computing technical indicators",
        "Computing fundamental metrics",
        "Running statistical models & Monte Carlo",
        "Exporting CSV & running ML predictions",
        "Running LLM analysis (5 teams, 13 agents)",
        "Rendering terminal report",
        "Saving Markdown report & charts",
        "Generating PDF report",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
        transient=True,   # Clears the progress bar after completion
    ) as progress:

        task = progress.add_task("", total=len(steps))

        # Step 1 — Price data
        progress.update(task, description=f"[1/{len(steps)}] {steps[0]}")
        try:
            price_df, info = fetch_stock_data(ticker)
        except ValueError as e:
            console.print(f"\n[bold red]ERROR:[/bold red] {e}")
            sys.exit(1)
        progress.advance(task)

        # Step 2 — Analyst data
        progress.update(task, description=f"[2/{len(steps)}] {steps[1]}")
        analyst_data = fetch_analyst_data(ticker)
        progress.advance(task)

        # Fetch options data (supplementary — non-fatal)
        try:
            options_data = fetch_options_data(ticker)
        except Exception:
            options_data = {"put_call_ratio": None, "options_iv_avg": None}

        # Step 3 — Technical indicators
        progress.update(task, description=f"[3/{len(steps)}] {steps[2]}")
        technical = compute_all_technicals(price_df)
        progress.advance(task)

        # Step 4 — Fundamental metrics
        progress.update(task, description=f"[4/{len(steps)}] {steps[3]}")
        fundamental = compute_all_fundamentals(info)
        progress.advance(task)

        # Step 5 — Statistical models
        progress.update(task, description=f"[5/{len(steps)}] {steps[4]}")
        statistical = compute_all_statistics(price_df, ticker)
        progress.advance(task)

        # Create report directory early so CSV export can use it below
        report_dir = create_report_dir(ticker)

        # Step 6 — Export CSV, score sentiment & run ML (before LLM so Judge has signal context)
        progress.update(task, description=f"[6/{len(steps)}] {steps[5]}")
        try:
            headlines = analyst_data.get("news", [])
            sentiment = score_headlines(headlines)
            try:
                # Archive today's sentiment so a real per-day history
                # accumulates for future leak-free use as an ML feature.
                save_sentiment_history(ticker, sentiment)
            except Exception:
                pass
            csv_path  = export_features_csv(
                ticker, price_df, technical, info, sentiment, report_dir
            )
            trainer_result = load_or_train(ticker, csv_path, force_retrain=retrain)
            ml_result = ml_predict(csv_path, trainer_result)
            ml_result["sentiment"] = sentiment
        except Exception as ml_err:
            console.log(f"[yellow]ML pipeline error: {ml_err}[/yellow]")
            ml_result = {
                "clf_5d_prob_up": None, "clf_21d_prob_up": None,
                "clf_5d_direction": "N/A", "clf_21d_direction": "N/A",
                "reg_5d_return_pct": None, "reg_21d_return_pct": None,
                "metrics": {}, "trained_on_rows": 0,
                "status": "error", "sentiment": {"overall_score": 0.0, "label": "NEUTRAL", "summary": ""},
            }
        progress.advance(task)

        # ── Auto-backfill: bootstrap feedback loop if insufficient resolved rows ─ #
        # Runs once per ticker (after ML training) when the feedback CSV is thin.
        # Uses test-split rows from features.csv so no data leakage occurs.
        try:
            if trainer_result and needs_backfill(ticker):
                bf_count = backfill_historical_predictions(
                    ticker, trainer_result, csv_path, price_df
                )
                if bf_count > 0:
                    console.log(
                        f"[dim]Feedback backfill: wrote {bf_count} historical rows for {ticker} "
                        f"(test-split replay — bootstrapping accuracy stats).[/dim]"
                    )
                    # Rebuild calibration table now that we have more resolved rows
                    build_calibration_table(ticker)
        except Exception as bf_err:
            console.log(f"[dim yellow]Backfill error (non-fatal): {bf_err}[/dim yellow]")

        # Step 7 — LLM Analysis (ml_result passed so Judge can use ML signals)
        progress.update(task, description=f"[7/{len(steps)}] {steps[6]}")
        if skip_llm or not LLM_ENABLED:
            if skip_llm:
                console.log("[dim]LLM analysis skipped via --no-llm flag — running rule-based analysis.[/dim]")
            try:
                llm_result = run_rule_based_analysis(
                    info, technical, fundamental, statistical, analyst_data, ml_result,
                    accuracy_context=accuracy_context,
                )
            except Exception as rbj_err:
                console.log(f"[yellow]Rule-based analysis error: {rbj_err}[/yellow]")
                llm_result = {
                    "llm_available":      False,
                    "recommendation":     "N/A — LLM disabled",
                    "confidence":         "N/A",
                    "overall_short_term": "N/A",
                    "overall_long_term":  "N/A",
                    "target_prices":      {},
                    "trade_direction":    "N/A",
                    "entry_price":        None,
                    "exit_price":         None,
                    "stop_loss":          None,
                    "position_size_pct":  None,
                    "key_bull_case":      [],
                    "key_bear_case":      [],
                    "key_risks":          [],
                    "catalysts":          [],
                    "summary":            "Rule-based analysis failed. See logs.",
                    "technical_verdict":  "See technical section.",
                    "fundamental_verdict":"See fundamental section.",
                    "valuation_verdict":  "See statistical section.",
                    "sentiment_verdict":  "N/A",
                    "macro_verdict":      "N/A",
                    "agents":             {},
                    "providers":          {},
                    "rule_based_judge":   None,
                    "description":        build_company_description(info, analyst_data),
                }
        else:
            llm_result = get_llm_analysis(
                ticker, info, technical, fundamental, statistical, analyst_data, ml_result,
                accuracy_context=accuracy_context,
                accuracy_context_map=accuracy_context_map,
            )
            # Always attach the rule-based judge so it can be displayed as a
            # standalone "second opinion" panel even when the LLM ran successfully.
            try:
                _rbj = run_rule_based_judge(
                    info, technical, fundamental, statistical, analyst_data, ml_result
                )
                llm_result["rule_based_judge"] = _rbj
            except Exception as rbj_err:
                console.log(f"[yellow]Rule-based judge (supplemental) error: {rbj_err}[/yellow]")
                llm_result.setdefault("rule_based_judge", None)
        progress.advance(task)

        # Step 8 — Terminal rendering
        progress.update(task, description=f"[8/{len(steps)}] {steps[7]}")
        # We need to finish progress bar before printing the full report
        progress.advance(task)

        # Step 9 — Save Markdown report + charts
        progress.update(task, description=f"[9/{len(steps)}] {steps[8]}")
        report_path = generate_markdown_report(
            ticker, info, technical, fundamental, statistical,
            analyst_data, llm_result, ml_result, report_dir,
            accuracy_context=accuracy_context,
        )
        progress.advance(task)

        # Step 10 — Generate PDF report
        progress.update(task, description=f"[10/{len(steps)}] {steps[9]}")
        try:
            pdf_path = generate_pdf_report(
                ticker, info, technical, fundamental, statistical,
                analyst_data, llm_result, report_dir
            )
        except Exception as pdf_err:  # PDF generation should never abort the run
            import traceback
            console.log(f"[yellow]PDF generation error: {pdf_err}[/yellow]")
            console.log(f"[dim]{traceback.format_exc()}[/dim]")
            pdf_path = None
        progress.advance(task)

    # ---- Print terminal report (after progress bar clears) ---- #
    display_full_report(
        ticker       = ticker,
        info         = info,
        technical    = technical,
        fundamental  = fundamental,
        statistical  = statistical,
        analyst_data = analyst_data,
        llm          = llm_result,
        ml_result    = ml_result,
        report_path  = report_path,
        accuracy_context = accuracy_context,
    )

    # ---- Save feedback prediction — one row per active provider ---- #
    try:
        if mode_str == "llm" and llm_result.get("providers"):
            for pname, presult in llm_result["providers"].items():
                # Skip providers that failed entirely (no LLM output produced)
                if not presult.get("llm_available", True):
                    continue
                save_feedback(
                    ticker       = ticker,
                    mode         = mode_str,
                    provider     = pname,
                    price_df     = price_df,
                    info         = info,
                    llm_result   = presult,
                    ml_result    = ml_result,
                    analyst_data = analyst_data,
                    options_data = options_data,
                )
        else:
            save_feedback(
                ticker       = ticker,
                mode         = mode_str,
                provider     = "rule_based",
                price_df     = price_df,
                info         = info,
                llm_result   = llm_result,
                ml_result    = ml_result,
                analyst_data = analyst_data,
                options_data = options_data,
            )
    except Exception as fb_save_err:
        console.log(f"[dim yellow]Feedback save error (non-fatal): {fb_save_err}[/dim yellow]")

    if pdf_path:
        console.print(f"\n[bold green]PDF report saved:[/bold green] {pdf_path}")
    else:
        console.print("\n[yellow]PDF report could not be generated (see log above).[/yellow]")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def _read_tickers_file(path: str) -> list[str]:
    """
    Parse a watchlist text file into a list of ticker strings.

    Rules applied during parsing:
      • Strip leading/trailing whitespace from every line.
      • Skip lines that start with '#' (comment lines).
      • Skip blank lines.
      • Normalise each ticker to uppercase so 'aapl' and 'AAPL' both work.

    Parameters
    ----------
    path : str
        Filesystem path to the watchlist file (e.g. 'tickers.txt').

    Returns
    -------
    list[str]
        Ordered, de-whitespaced ticker symbols ready for run_pipeline().

    Raises
    ------
    SystemExit
        If the file cannot be opened (wrong path / no permission).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        console.print(f"[bold red]ERROR:[/bold red] Cannot open watchlist file: {exc}")
        sys.exit(1)

    tickers: list[str] = []
    for line in lines:
        # Drop inline comments (watchlist.txt entries carry provenance
        # like "QUBT  # discovered 2026-07-04 via day_gainers"), then
        # strip whitespace and ignore comment / blank lines.
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        tickers.append(token.upper())

    if not tickers:
        console.print(
            f"[bold yellow]WARNING:[/bold yellow] No tickers found in '{path}'. "
            "Check the file contains at least one non-comment line."
        )
        sys.exit(0)

    return tickers


if __name__ == "__main__":
    args = parse_args()

    # ── --retrain-global mode: train pooled model and exit ─────── #
    if args.retrain_global:
        from ml.global_trainer import train_global_models
        console.print("\n[bold cyan]FinBot — Training pooled cross-ticker model[/bold cyan]\n")
        with console.status("[dim]Stacking features and running walk-forward CV…[/dim]"):
            gres = train_global_models()
        if gres["status"] == "trained":
            m = gres["metrics"]
            console.print(
                f"[bold green]Global model trained[/bold green] on "
                f"{gres['rows']} pooled rows from {len(gres['tickers'])} ticker(s): "
                f"{', '.join(gres['tickers'])}"
            )
            for h in (5, 21):
                acc = m.get(f"clf_{h}d_accuracy")
                std = m.get(f"clf_{h}d_accuracy_std")
                console.print(f"  {h:>2}d direction CV accuracy: {acc} (±{std})")
        else:
            console.print(
                "[yellow]Not enough pooled data to train — run a few normal "
                "analyses first so reports/*/features.csv exist.[/yellow]"
            )
        sys.exit(0)

    # ── --backtest mode: validate the technical lens and exit ──── #
    if args.backtest:
        from analysis.backtester import run_backtest
        bt_tickers: list[str] = []
        if args.ticker:
            bt_tickers = [args.ticker.upper().strip()]
        elif args.file:
            bt_tickers = _read_tickers_file(args.file)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for fname in ("tickers.txt", "watchlist.txt"):
                path = os.path.join(base_dir, fname)
                if os.path.isfile(path):
                    try:
                        bt_tickers += _read_tickers_file(path)
                    except SystemExit:
                        pass    # empty watchlist etc. — not fatal here
        if not bt_tickers and not args.backtest_broad:
            console.print("[yellow]No tickers to backtest.[/yellow]")
            sys.exit(0)
        run_backtest(console, bt_tickers, broad=args.backtest_broad)
        sys.exit(0)

    # ── --discover mode: scan market for new candidates and exit ─ #
    if args.discover:
        from discovery.screener import run_discovery
        run_discovery(
            console  = console,
            top_n    = args.discover_top,
            min_cap  = args.discover_min_cap,
        )
        sys.exit(0)

    # ── --resolve mode: settle pending predictions and exit ────── #
    if args.resolve:
        from feedback.resolver import resolve_all as _resolve_all
        target_tickers: list[str] = []
        if args.ticker:
            target_tickers = [args.ticker.upper().strip()]
        elif args.file:
            target_tickers = _read_tickers_file(args.file)
        else:
            # Default: resolve all tickers in tickers.txt if it exists
            default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.txt")
            if os.path.isfile(default_file):
                target_tickers = _read_tickers_file(default_file)

        if not target_tickers:
            console.print("[yellow]No tickers to resolve.[/yellow]")
            sys.exit(0)

        console.print(f"\n[bold cyan]FinBot — Resolving feedback for {len(target_tickers)} ticker(s)[/bold cyan]\n")
        total = 0
        for t in target_tickers:
            n = 0
            try:
                n = __import__("feedback.resolver", fromlist=["resolve_pending"]).resolve_pending(t)
            except Exception as e:
                console.print(f"  [red]{t}[/red]: error — {e}")
                continue
            colour = "green" if n > 0 else "dim"
            console.print(f"  [{colour}]{t:>10}[/{colour}]  {n} outcome(s) resolved")
            total += n
        console.print(f"\n[bold green]Done.[/bold green] {total} total outcome(s) resolved across all tickers.\n")
        sys.exit(0)

    # ── --backfill mode: bootstrap feedback from features.csv ──── #
    if args.backfill:
        target_tickers: list[str] = []
        if args.ticker:
            target_tickers = [args.ticker.upper().strip()]
        elif args.file:
            target_tickers = _read_tickers_file(args.file)
        else:
            default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.txt")
            if os.path.isfile(default_file):
                target_tickers = _read_tickers_file(default_file)

        if not target_tickers:
            console.print("[yellow]No tickers to backfill.[/yellow]")
            sys.exit(0)

        from feedback.backfiller import backfill_historical_predictions as _backfill
        from data.stock_fetcher import fetch_stock_data as _fetch
        from ml.trainer import load_or_train as _load_or_train
        import glob

        console.print(f"\n[bold cyan]FinBot — Backfilling feedback for {len(target_tickers)} ticker(s)[/bold cyan]\n")
        total_bf = 0
        for t in target_tickers:
            try:
                # Find the most recent features.csv for this ticker
                pattern = os.path.join("reports", f"{t}_*", "features.csv")
                candidates = sorted(glob.glob(pattern))
                if not candidates:
                    console.print(f"  [yellow]{t:>10}[/yellow]  no features.csv found — run a normal analysis first")
                    continue
                csv_path = candidates[-1]

                # Load price history and models
                price_df, _ = _fetch(t)
                trainer_result = _load_or_train(t, csv_path)
                if trainer_result.get("status") == "insufficient_data":
                    console.print(f"  [yellow]{t:>10}[/yellow]  insufficient training data")
                    continue

                n = _backfill(t, trainer_result, csv_path, price_df)
                build_calibration_table(t)
                colour = "green" if n > 0 else "dim"
                console.print(f"  [{colour}]{t:>10}[/{colour}]  {n} historical row(s) backfilled")
                total_bf += n
            except Exception as e:
                console.print(f"  [red]{t}[/red]: error — {e}")
                continue
        console.print(f"\n[bold green]Done.[/bold green] {total_bf} total row(s) backfilled.\n")
        sys.exit(0)

    if args.ticker:
        # ── Single-ticker mode ──────────────────────────────────── #
        # Standard usage: `python main.py --ticker AAPL`
        run_pipeline(ticker=args.ticker, skip_llm=args.no_llm, retrain=args.retrain)

    else:
        # ── Batch / watchlist mode ──────────────────────────────── #
        # Usage: `python main.py --file tickers.txt [--no-llm] [--retrain]`
        #
        # The pipeline is run sequentially for each ticker in the file.
        # Per-ticker errors are caught and logged so a bad symbol or a
        # transient network failure does not abort the entire batch —
        # the run simply continues with the next ticker.
        tickers = _read_tickers_file(args.file)

        console.print(
            f"[bold cyan]FinBot batch run[/bold cyan] — "
            f"{len(tickers)} ticker(s) from [yellow]{args.file}[/yellow]\n"
        )

        # ── Prewarm the shared data cache in parallel ───────────────── #
        # Fetch each ticker's history/info/news/earnings (plus VIX and
        # SPY once for the whole batch) into data/market_data.py's cache
        # so the sequential pipeline below runs from memory. Worker count
        # is deliberately low to stay clear of Yahoo rate limits.
        try:
            from concurrent.futures import ThreadPoolExecutor
            from data import market_data
            from config import HISTORY_PERIOD

            with console.status("[dim]Prewarming market data cache…[/dim]"):
                market_data.get_spy()
                market_data.get_vix_latest()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    for t in tickers:
                        pool.submit(market_data.prewarm_symbol, t, HISTORY_PERIOD)
        except Exception as warm_err:
            console.log(f"[dim yellow]Cache prewarm error (non-fatal): {warm_err}[/dim yellow]")

        results: dict[str, str] = {}  # ticker → "ok" | error message

        for idx, ticker in enumerate(tickers, start=1):
            console.rule(
                f"[bold]{idx}/{len(tickers)}[/bold]  {ticker}",
                style="cyan",
            )
            try:
                run_pipeline(
                    ticker   = ticker,
                    skip_llm = args.no_llm,
                    retrain  = args.retrain,
                )
                results[ticker] = "ok"
            except SystemExit:
                # run_pipeline calls sys.exit(1) on an invalid ticker —
                # intercept it here so the batch does not terminate early.
                results[ticker] = "invalid ticker / no data"
                console.print(
                    f"[yellow]Skipping {ticker} — no price data found.[/yellow]\n"
                )
            except Exception as exc:  # noqa: BLE001
                # Catch any unexpected exception so the batch keeps running.
                results[ticker] = str(exc)
                console.print(
                    f"[bold red]ERROR[/bold red] processing {ticker}: {exc}\n"
                )

        # ── Final summary table ─────────────────────────────────── #
        console.rule("[bold cyan]Batch Summary[/bold cyan]", style="cyan")
        for sym, status in results.items():
            colour = "green" if status == "ok" else "red"
            console.print(f"  [{colour}]{sym:>10}[/{colour}]  {status}")
        console.print()
