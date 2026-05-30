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
import sys
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# ---- Internal modules ---- #
from data.stock_fetcher    import fetch_stock_data
from data.analyst_fetcher  import fetch_analyst_data
from analysis.technical    import compute_all_technicals
from analysis.fundamental  import compute_all_fundamentals
from analysis.statistical  import compute_all_statistics
from analysis.llm_analysis    import get_llm_analysis, build_company_description
from analysis.rule_based_judge import run_rule_based_judge, run_rule_based_analysis
from output.terminal_display import display_full_report
from output.report_generator import create_report_dir, generate_markdown_report
from output.pdf_generator    import generate_pdf_report
from config import LLM_ENABLED
from data.csv_exporter      import export_features_csv
from data.sentiment_fetcher import score_headlines
from ml.trainer    import load_or_train
from ml.predictor  import predict as ml_predict

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
    parser.add_argument(
        "--ticker", "-t",
        type    = str,
        required= True,
        help    = "Yahoo Finance ticker symbol (e.g. AAPL, MSFT, TSLA, 9988.HK)",
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
    return parser.parse_args()


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

    console.print()
    console.print(f"[bold cyan]FinBot[/bold cyan] — Analysing [bold yellow]{ticker}[/bold yellow] ...")
    console.print()

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

        # Step 7 — LLM Analysis (ml_result passed so Judge can use ML signals)
        progress.update(task, description=f"[7/{len(steps)}] {steps[6]}")
        if skip_llm or not LLM_ENABLED:
            if skip_llm:
                console.log("[dim]LLM analysis skipped via --no-llm flag — running rule-based analysis.[/dim]")
            try:
                llm_result = run_rule_based_analysis(
                    info, technical, fundamental, statistical, analyst_data, ml_result
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
                ticker, info, technical, fundamental, statistical, analyst_data, ml_result
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
            analyst_data, llm_result, ml_result, report_dir
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
    )

    if pdf_path:
        console.print(f"\n[bold green]PDF report saved:[/bold green] {pdf_path}")
    else:
        console.print("\n[yellow]PDF report could not be generated (see log above).[/yellow]")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(ticker=args.ticker, skip_llm=args.no_llm, retrain=args.retrain)
