"""
output/report_generator.py — Markdown report and chart generation for FinBot.

Builds a structured Markdown (.md) file with three embedded PNG charts:
  1. price_chart.png       — 2-year Close price with SMA-20/50/200 overlay
  2. technical_chart.png   — RSI subplot + MACD histogram subplot
  3. monte_carlo_chart.png — GBM simulation fan chart (P10/median/P90 bands)

All files are saved under:  reports/{TICKER}_{YYYY-MM-DD}/
"""

import os
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import REPORTS_DIR, CHART_DPI, PREDICTION_HORIZONS


# ------------------------------------------------------------------ #
# Directory management
# ------------------------------------------------------------------ #

def create_report_dir(ticker: str) -> str:
    """
    Create the report output directory and return its path.
    Path: reports/{TICKER}_{YYYY-MM-DD}/
    """
    date_str  = datetime.date.today().isoformat()
    dir_name  = f"{ticker.upper()}_{date_str}"
    dir_path  = os.path.join(REPORTS_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


# ------------------------------------------------------------------ #
# Chart 1: Price + Moving Averages
# ------------------------------------------------------------------ #

def _save_price_chart(technical: dict, dir_path: str, ticker: str) -> str:
    """
    Two-panel chart:
      Top:    Close price with SMA-20 (blue), SMA-50 (orange), SMA-200 (red)
              Plus Bollinger Bands shaded region
      Bottom: Volume bars
    """
    df    = technical["df"]
    dates = df.index

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.suptitle(f"{ticker} — Price & Moving Averages", fontsize=14, fontweight="bold")

    # Close price
    ax1.plot(dates, df["Close"],  color="white",  linewidth=1.2, label="Close", zorder=3)

    # Bollinger band shading
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        ax1.fill_between(dates, df["BB_Lower"], df["BB_Upper"],
                         alpha=0.10, color="cyan", label="Bollinger Bands")
        ax1.plot(dates, df["BB_Upper"],  color="cyan", linewidth=0.5, alpha=0.5)
        ax1.plot(dates, df["BB_Lower"],  color="cyan", linewidth=0.5, alpha=0.5)

    # Moving averages
    if "SMA_20"  in df.columns: ax1.plot(dates, df["SMA_20"],  color="#4FC3F7", linewidth=1.0, label="SMA-20")
    if "SMA_50"  in df.columns: ax1.plot(dates, df["SMA_50"],  color="#FFB74D", linewidth=1.0, label="SMA-50")
    if "SMA_200" in df.columns: ax1.plot(dates, df["SMA_200"], color="#EF9A9A", linewidth=1.2, label="SMA-200")

    ax1.set_facecolor("#1a1a2e")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.grid(alpha=0.15)

    # Volume
    vol_colors = ["#4CAF50" if c >= o else "#F44336"
                  for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(dates, df["Volume"], color=vol_colors, alpha=0.7, width=0.8)
    ax2.set_facecolor("#1a1a2e")
    ax2.set_ylabel("Volume")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax2.grid(alpha=0.10)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate(rotation=30)
    fig.patch.set_facecolor("#0d0d1a")

    path = os.path.join(dir_path, "price_chart.png")
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
# Chart 2: RSI + MACD Technical Subplots
# ------------------------------------------------------------------ #

def _save_technical_chart(technical: dict, dir_path: str, ticker: str) -> str:
    """
    Three-panel chart:
      Top:    Close price (simplified)
      Middle: RSI with 30/70 overbought/oversold lines
      Bottom: MACD histogram with MACD and Signal line overlay
    """
    df    = technical["df"]
    dates = df.index

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 1]}, sharex=True
    )
    fig.suptitle(f"{ticker} — Technical Indicators (RSI & MACD)", fontsize=14, fontweight="bold")

    # Top: Close price
    ax1.plot(dates, df["Close"].dropna(), color="white", linewidth=1.0, label="Close")
    ax1.set_facecolor("#1a1a2e")
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=8, framealpha=0.3)
    ax1.grid(alpha=0.12)

    # Middle: RSI
    rsi_valid = df["RSI"].dropna()
    ax2.plot(rsi_valid.index, rsi_valid, color="#AB47BC", linewidth=1.2, label="RSI(14)")
    ax2.axhline(70, color="#EF5350", linestyle="--", linewidth=0.8, alpha=0.8, label="Overbought (70)")
    ax2.axhline(30, color="#66BB6A", linestyle="--", linewidth=0.8, alpha=0.8, label="Oversold (30)")
    ax2.axhline(50, color="gray",    linestyle=":",  linewidth=0.6, alpha=0.5)
    ax2.fill_between(rsi_valid.index, rsi_valid, 70, where=(rsi_valid >= 70), alpha=0.2, color="#EF5350")
    ax2.fill_between(rsi_valid.index, rsi_valid, 30, where=(rsi_valid <= 30), alpha=0.2, color="#66BB6A")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_facecolor("#1a1a2e")
    ax2.legend(fontsize=7, framealpha=0.3)
    ax2.grid(alpha=0.10)

    # Bottom: MACD histogram + lines
    macd_hist  = df["MACD_Hist"].dropna()
    macd_line  = df["MACD_Line"].dropna()
    macd_sig   = df["MACD_Signal"].dropna()
    hist_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in macd_hist]
    ax3.bar(macd_hist.index, macd_hist, color=hist_colors, alpha=0.7, width=0.8, label="Histogram")
    ax3.plot(macd_line.index,  macd_line,  color="#29B6F6", linewidth=1.0, label="MACD")
    ax3.plot(macd_sig.index,   macd_sig,   color="#FF7043", linewidth=1.0, label="Signal")
    ax3.axhline(0, color="white", linewidth=0.5, alpha=0.4)
    ax3.set_ylabel("MACD")
    ax3.set_facecolor("#1a1a2e")
    ax3.legend(fontsize=7, framealpha=0.3)
    ax3.grid(alpha=0.10)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate(rotation=30)
    fig.patch.set_facecolor("#0d0d1a")

    path = os.path.join(dir_path, "technical_chart.png")
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
# Chart 3: Monte Carlo Fan Chart
# ------------------------------------------------------------------ #

def _save_montecarlo_chart(statistical: dict, dir_path: str, ticker: str) -> str:
    """
    Fan chart showing the envelope of 1 000 Monte Carlo GBM paths.
    Plots:
      • Filled band between P10 and P90 (uncertainty range)
      • Highlighted median path
      • 50 individual sample paths (thin, semi-transparent)
      • Vertical dashed lines at each prediction horizon
    """
    mc      = statistical.get("monte_carlo", {})
    paths   = mc.get("paths")
    preds   = mc.get("predictions", {})
    params  = mc.get("params", {})

    if paths is None:
        return ""

    max_h  = max(PREDICTION_HORIZONS.values())
    x_days = np.arange(max_h + 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"{ticker} — Monte Carlo GBM Simulation ({params.get('n_paths',1000):,} paths)",
        fontsize=14, fontweight="bold"
    )

    # Plot 50 random sample paths (thin, transparent)
    sample_idx = np.random.choice(paths.shape[0], size=min(50, paths.shape[0]), replace=False)
    for idx in sample_idx:
        ax.plot(x_days, paths[idx], color="#90CAF9", linewidth=0.3, alpha=0.25)

    # Percentile bands at every step
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    ax.fill_between(x_days, p10, p90, alpha=0.25, color="#42A5F5", label="P10–P90 range")
    ax.plot(x_days, p90,     color="#66BB6A", linewidth=1.2, linestyle="--", label="P90 (Bull)")
    ax.plot(x_days, p50,     color="#FFEE58", linewidth=2.0, label="Median (P50)")
    ax.plot(x_days, p10,     color="#EF5350", linewidth=1.2, linestyle="--", label="P10 (Bear)")

    # Mark prediction horizon lines
    horizon_labels = {v: k for k, v in PREDICTION_HORIZONS.items()}
    for h in PREDICTION_HORIZONS.values():
        ax.axvline(h, color="white", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.text(h + 1, ax.get_ylim()[1] * 0.99,
                horizon_labels.get(h, str(h)),
                color="white", fontsize=7, va="top", alpha=0.7)

    ax.set_xlabel("Trading Days from Today")
    ax.set_ylabel("Price (USD)")
    ax.set_facecolor("#1a1a2e")
    ax.legend(fontsize=9, framealpha=0.3, loc="upper left")
    ax.grid(alpha=0.12)
    fig.patch.set_facecolor("#0d0d1a")

    path = os.path.join(dir_path, "monte_carlo_chart.png")
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
# Markdown Report Builder
# ------------------------------------------------------------------ #

def _fmt(val, prefix="", suffix="", decimals=2, null="N/A") -> str:
    """Safe formatter that returns null string when val is None."""
    if val is None:
        return null
    try:
        return f"{prefix}{val:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def generate_markdown_report(
    ticker: str,
    info: dict,
    technical: dict,
    fundamental: dict,
    statistical: dict,
    analyst_data: dict,
    llm: dict,
    ml_result: dict | None,
    dir_path: str,
) -> str:
    """
    Assemble the full Markdown report and save it to dir_path/report.md.

    Returns the absolute path to the saved file.
    """

    # ---- Generate charts first ------------------------------------ #
    chart_price = _save_price_chart(technical, dir_path, ticker)
    chart_tech  = _save_technical_chart(technical, dir_path, ticker)
    chart_mc    = _save_montecarlo_chart(statistical, dir_path, ticker)

    # ---- shortcuts ----------------------------------------------- #
    name       = info.get("shortName", ticker)
    price      = info.get("currentPrice", "N/A")
    sector     = info.get("sector", "N/A")
    industry   = info.get("industry", "N/A")
    mcap       = info.get("marketCap")
    mcap_str   = f"${mcap:,.0f}" if mcap else "N/A"
    hi52       = info.get("fiftyTwoWeekHigh", "N/A")
    lo52       = info.get("fiftyTwoWeekLow",  "N/A")
    run_date   = datetime.date.today().isoformat()

    pe     = fundamental.get("pe", {})
    peg    = fundamental.get("peg", {})
    graham = fundamental.get("graham", {})
    dcf    = fundamental.get("dcf", {})
    pb_div = fundamental.get("pb_div", {})

    vol     = statistical.get("volatility", {})
    beta    = statistical.get("beta", {})
    sharpe  = statistical.get("sharpe", {})
    sortino = statistical.get("sortino", {})
    drawdown= statistical.get("drawdown", {})
    reg     = statistical.get("regression", {})
    mc      = statistical.get("monte_carlo", {})

    tech_latest  = technical.get("latest", {})
    tech_signals = technical.get("signals", {})
    pivot        = technical.get("pivot_levels", {})
    fibs         = technical.get("fibonacci", {})

    rec_key = analyst_data.get("recommendation_key", "N/A").upper()
    pt      = analyst_data.get("price_target") or {}
    news    = analyst_data.get("news", [])

    sigma_annual_pct = (
        f"{vol.get('sigma_annual', 0) * 100:.2f}%"
        if isinstance(vol.get("sigma_annual"), float) else "N/A"
    )

    # ---- BUILD MARKDOWN STRING ------------------------------------ #
    lines = []
    A = lines.append   # shorthand

    A(f"# FinBot Analysis Report: {name} ({ticker})")
    A(f"\n> **Generated:** {run_date}  |  **Current Price:** ${price}  |  **Sector:** {sector}\n")
    A("---\n")

    # Overview
    A("## Company Overview\n")
    A("| Field | Value |")
    A("|---|---|")
    A(f"| **Ticker** | {ticker} |")
    A(f"| **Company** | {name} |")
    A(f"| **Sector** | {sector} |")
    A(f"| **Industry** | {industry} |")
    A(f"| **Market Cap** | {mcap_str} |")
    A(f"| **Current Price** | ${price} |")
    A(f"| **52-Week High** | ${hi52} |")
    A(f"| **52-Week Low** | ${lo52} |")
    A("")

    # Price Chart
    A("## Price History & Moving Averages\n")
    A(f"![Price Chart](price_chart.png)\n")

    # Basic Description (from yfinance — no LLM)
    A("## Basic Description\n")
    desc = llm.get("description") or {}
    if desc.get("business_overview"):
        name     = desc.get("company_name", info.get("shortName", ""))
        sector   = desc.get("sector", "N/A")
        industry = desc.get("industry", "N/A")
        country  = desc.get("country", "N/A")
        website  = desc.get("website", "")
        employees = desc.get("employees", "N/A")
        founded  = desc.get("founding_year", "N/A")
        A(f"**{name}** | Sector: {sector} | Industry: {industry}  ")
        A(f"Country: {country} | Employees: {employees} | Est. ~{founded}  ")
        if website and website != "N/A": A(f"{website}\n")
        A("")
        A(f"{desc['business_overview']}\n")
        desc_news = desc.get("recent_news", [])
        if desc_news:
            A("\n**Recent News:**\n")
            for h in desc_news[:5]: A(f"- {h}")
            A("")
    else:
        summary_text = info.get("longBusinessSummary", "")
        if summary_text:
            A(f"{summary_text}\n")
        else:
            A("> Company description not available.\n")

    # Technical
    A("## Technical Analysis\n")
    A("### Indicator Values\n")
    A("| Indicator | Value | Signal |")
    A("|---|---|---|")
    A(f"| RSI (14) | {tech_latest.get('RSI','N/A')} | {tech_signals.get('RSI','')} |")
    A(f"| MACD Line | {tech_latest.get('MACD_Line','N/A')} | {tech_signals.get('MACD','')} |")
    A(f"| MACD Signal | {tech_latest.get('MACD_Signal','N/A')} | |")
    A(f"| MACD Histogram | {tech_latest.get('MACD_Hist','N/A')} | |")
    A(f"| SMA-20 | {tech_latest.get('SMA_20','N/A')} | |")
    A(f"| SMA-50 | {tech_latest.get('SMA_50','N/A')} | {tech_signals.get('MA_Trend','')} |")
    A(f"| SMA-200 | {tech_latest.get('SMA_200','N/A')} | |")
    A(f"| EMA-12 | {tech_latest.get('EMA_12','N/A')} | |")
    A(f"| EMA-26 | {tech_latest.get('EMA_26','N/A')} | |")
    A(f"| Bollinger Upper | {tech_latest.get('BB_Upper','N/A')} | {tech_signals.get('Bollinger','')} |")
    A(f"| Bollinger Middle | {tech_latest.get('BB_Middle','N/A')} | |")
    A(f"| Bollinger Lower | {tech_latest.get('BB_Lower','N/A')} | |")
    A(f"| BB %B | {tech_latest.get('BB_PctB','N/A')} | |")
    A("")
    A("### Fibonacci Retracement Levels\n")
    A("| Level | Price |")
    A("|---|---|")
    for label, fp in fibs.items():
        A(f"| {label} | ${fp} |")
    A("")
    A(f"**Resistance pivots:** {', '.join('$'+str(r) for r in pivot.get('resistance_levels',[]))  or 'N/A'}\n")
    A(f"**Support pivots:**    {', '.join('$'+str(s) for s in pivot.get('support_levels',[]))  or 'N/A'}\n")
    A(f"![Technical Chart](technical_chart.png)\n")

    # Fundamental
    A("## Fundamental Analysis\n")
    A("| Metric | Value | Signal / Interpretation |")
    A("|---|---|---|")
    A(f"| Trailing P/E | {pe.get('trailing_pe','N/A')} | {pe.get('trailing_signal','')} |")
    A(f"| Forward P/E | {pe.get('forward_pe','N/A')} | {pe.get('forward_signal','')} |")
    A(f"| PEG Ratio | {peg.get('peg','N/A')} | {peg.get('signal','')} |")
    A(f"| Graham Number | ${graham.get('graham_number','N/A')} | {graham.get('signal','')} |")
    A(f"| DCF Intrinsic Value | ${dcf.get('dcf_value','N/A')} | {dcf.get('signal','')} |")
    A(f"| DCF Growth Rate | {dcf.get('growth_rate_used','N/A')} | WACC: {dcf.get('wacc_used','N/A')} |")
    A(f"| Price-to-Book | {pb_div.get('price_to_book','N/A')} | {pb_div.get('pb_signal','')} |")
    A(f"| Dividend Yield | {pb_div.get('dividend_yield','N/A')}% | {pb_div.get('dividend_signal','')} |")
    A("")

    # Statistical
    A("## Statistical Analysis\n")
    A("### Risk Metrics\n")
    A("| Metric | Value | Signal |")
    A("|---|---|---|")
    A(f"| Annual Volatility (σ) | {sigma_annual_pct} | {vol.get('label','')} |")
    A(f"| Beta (vs S&P 500) | {beta.get('beta','N/A')} | {beta.get('label','')} |")
    A(f"| Sharpe Ratio | {sharpe.get('sharpe','N/A')} | {sharpe.get('signal','')} |")
    A(f"| Historical Annual Return | {sharpe.get('annual_return','N/A')} | |")
    A(f"| Sortino Ratio | {sortino.get('sortino','N/A')} | {sortino.get('signal','')} |")
    A(f"| Max Drawdown | {drawdown.get('max_drawdown_pct','N/A')}% | |")
    A(f"| Calmar Ratio | {drawdown.get('calmar','N/A')} | {drawdown.get('signal','')} |")
    A("")

    # Linear Regression predictions
    reg_preds = reg.get("predictions", {})
    A("### Linear Regression Price Predictions\n")
    A(f"> R² = {reg.get('r_squared','N/A')} | Trend: {reg.get('trend_dir','N/A')}\n")
    A("| Horizon | Target Price | Expected Change |")
    A("|---|---|---|")
    for h_label, data in reg_preds.items():
        sign = "+" if data["change%"] >= 0 else ""
        A(f"| {h_label} | ${data['price']} | {sign}{data['change%']}% |")
    A("")

    # Monte Carlo predictions
    mc_preds = mc.get("predictions", {})
    mc_params = mc.get("params", {})
    mu_method = mc_params.get("mu_method", "historical")
    A(f"### Monte Carlo GBM Price Forecasts (1 000 Paths, drift: {mu_method})\n")
    A("| Horizon | Bear (P10) | Median (P50) | Bull (P90) |")
    A("|---|---|---|---|")
    for h_label, data in mc_preds.items():
        A(f"| {h_label} | ${data['p10']} ({data['p10_chg%']:+.1f}%) | "
          f"${data['median']} ({data['med_chg%']:+.1f}%) | "
          f"${data['p90']} ({data['p90_chg%']:+.1f}%) |")
    A("")
    A(f"![Monte Carlo Chart](monte_carlo_chart.png)\n")

    # Analyst Consensus
    A("## Analyst Consensus\n")
    A(f"- **Consensus Rating:** {rec_key}")
    A(f"- **Mean Price Target:** ${pt.get('mean','N/A')}")
    A(f"- **Low Price Target:**  ${pt.get('low','N/A')}")
    A(f"- **High Price Target:** ${pt.get('high','N/A')}")

    # Analyst upgrade/downgrade history
    recs = analyst_data.get("recommendations", [])
    if recs:
        # Sentiment summary
        recent_10 = recs[:10]
        n_up   = sum(1 for r in recent_10 if "Upgrade"    in r.get("action", ""))
        n_down = sum(1 for r in recent_10 if "Downgrade"  in r.get("action", ""))
        n_init = sum(1 for r in recent_10 if "Initiation" in r.get("action", ""))
        sentiment = (
            "Net Positive" if n_up > n_down else
            "Net Negative" if n_down > n_up else "Mixed / Neutral"
        )
        A(f"\n**Analyst Sentiment (last 10 actions):** {n_up} upgrade(s), "
          f"{n_down} downgrade(s), {n_init} new initiation(s) — **{sentiment}**\n")
        A("\n### Analyst Upgrade/Downgrade History\n")
        A("| Date | Firm | Action | From Grade | To Grade |")
        A("|---|---|---|---|---|")
        for r in recs[:20]:
            A(f"| {r['date']} | {r['firm']} | {r['action']} "
              f"| {r.get('from_grade','') or '—'} | {r.get('to_grade','') or 'N/A'} |")
        A("")
        A("<details>")
        A("<summary><b>Legend — Actions &amp; Grade Terminology</b></summary>\n")
        A("**Actions**\n")
        A("| Action | Meaning |")
        A("|---|---|")
        A("| **Upgrade** | Analyst raises the stock's rating (e.g. Hold → Buy). Bullish signal. |")
        A("| **Downgrade** | Analyst cuts the stock's rating (e.g. Buy → Neutral). Bearish signal. |")
        A("| **Initiation** | First-ever coverage by this firm — no prior grade exists. |")
        A("| **Maintained** | Analyst reaffirms the existing rating without any change. |")
        A("| **Reiterated** | Same as Maintained — a formal restatement of the prior rating. |")
        A("")
        A("**Grade Scale** _(terminology varies by brokerage)_\n")
        A("| Grade(s) | Meaning |")
        A("|---|---|")
        A("| Strong Buy · Conviction Buy · Top Pick | Highest conviction bullish call. |")
        A("| Buy · Outperform · Overweight · Positive | Bullish — expected to outperform the market. |")
        A("| Hold · Neutral · Market Perform | Neutral — analyst expects returns roughly in line with the broader market. |")
        A("| Equal Weight | Same as Hold/Neutral. Used by Morgan Stanley, Barclays and others. |")
        A("| **Sector Perform** | The stock is expected to perform **in line with its sector peers** — "
          "neither outperform nor underperform the sector average. "
          "Used by RBC, CIBC, and similar brokerages. Equivalent to Hold / Neutral / Market Perform. |")
        A("| Underperform · Underweight · Sector Underperform | Mild bearish — expected to lag the market. |")
        A("| Sell · Strong Sell | Bearish — analyst expects a material price decline. |")
        A("")
        A("**Columns**\n")
        A("| Column | Meaning |")
        A("|---|---|")
        A("| **From** | The rating held by the analyst *before* today's action. |")
        A("| **To Grade** | The *new* rating after today's action — this is the current live rating. |")
        A("</details>\n")

    if news:
        A("\n### Recent News Headlines\n")
        for item in news[:5]:
            A(f"- [{item['title']}]({item.get('link','')}) _{item.get('publisher','')}_")
    A("")

    # ML Predictions
    if ml_result and ml_result.get("status") not in (None, "insufficient_data", "error"):
        A("## ML Predictions (GradientBoosting)\n")

        sentiment   = ml_result.get("sentiment", {})
        sent_score  = sentiment.get("overall_score", 0.0)
        sent_label  = sentiment.get("label", "NEUTRAL")
        sent_summary = sentiment.get("summary", "")
        A(f"**Headline Sentiment:** {sent_label} (score {sent_score:+.2f})"
          + (f" — *{sent_summary}*" if sent_summary else "") + "\n")

        A("| Horizon | Direction | P(Up) | Expected Return |")
        A("|---|---|---|---|")
        for h, lbl in ((5, "5-Day (1 Week)"), (21, "21-Day (1 Month)")):
            direction = ml_result.get(f"clf_{h}d_direction", "N/A")
            prob_up   = ml_result.get(f"clf_{h}d_prob_up")
            ret_pct   = ml_result.get(f"reg_{h}d_return_pct")
            prob_str  = f"{prob_up:.1%}" if prob_up is not None else "N/A"
            ret_str   = f"{ret_pct:+.2f}%" if ret_pct is not None else "N/A"
            A(f"| {lbl} | {direction} | {prob_str} | {ret_str} |")
        A("")

        metrics      = ml_result.get("metrics", {})
        trained_rows = ml_result.get("trained_on_rows", 0)
        if metrics or trained_rows:
            A("**Model Quality Metrics**\n")
            A("| Metric | Value |")
            A("|---|---|")
            if trained_rows:
                A(f"| Training rows | {trained_rows} |")
            for h in (5, 21):
                acc = metrics.get(f"clf_{h}d_accuracy")
                mae = metrics.get(f"reg_{h}d_mae")
                if acc is not None: A(f"| Classifier {h}d accuracy | {acc:.1%} |")
                if mae is not None: A(f"| Regressor {h}d MAE | {mae:.4f} |")
            A("")
        A("> ⚠ *ML predictions are based on historical patterns and are not financial advice.*\n")

    # LLM Recommendation
    A("## AI Analyst Recommendation\n")
    if not llm.get("llm_available"):
        A(f"> ⚠ **LLM Not Available:** {llm.get('summary','')}\n")
        rbj = llm.get("rule_based_judge")
        if rbj:
            A("### Rule-Based Investment Judge\n")
            A("> *Deterministic four-lens analysis (no LLM required).*\n")
            composite = rbj.get("composite_score", "N/A")
            rec       = rbj.get("recommendation", "N/A")
            conf      = rbj.get("confidence", "N/A")
            worth     = rbj.get("worth_investing", "N/A")
            A(f"| Metric | Value |")
            A(f"|--------|-------|")
            A(f"| Composite Score | {composite}/100 |")
            A(f"| Recommendation | **{rec}** |")
            A(f"| Confidence | {conf} |")
            A(f"| Worth Investing | **{worth}** |")
            if rbj.get("veto_triggered"):
                A(f"| ⚠ Veto | {rbj.get('veto_reason','')} |")
            A("")
            for lens_key, lens_title in [
                ("fundamental", "Fundamental (Business Health)"),
                ("technical",   "Technical (Price/Timing)"),
                ("valuation",   "Valuation (Overpaying?)"),
                ("risk",        "Risk (Downside Exposure)"),
            ]:
                lens = rbj.get(lens_key, {})
                if not lens:
                    continue
                A(f"#### {lens_title}\n")
                A(f"Score: **{lens.get('score','N/A')}/100** | Grade: **{lens.get('grade','?')}** | Verdict: **{lens.get('verdict','N/A')}**\n")
                sigs = lens.get("signals", [])
                if sigs:
                    A("| Metric | Value | Assessment |")
                    A("|--------|-------|------------|")
                    for sig in sigs:
                        A(f"| {sig.get('metric','')} | {sig.get('value','N/A')} | {sig.get('label','N/A')} |")
                    A("")
            A(f"**Summary:** {rbj.get('summary','')}\n")
    else:
        providers = llm.get("providers", {})
        for provider_name, result in providers.items():
            A(f"### {provider_name}\n")

            # Surface chain-level errors (e.g. API failure, JSON parse error)
            if not result.get("llm_available", True) or result.get("recommendation") == "ERROR":
                err_msg = result.get("summary", "Unknown error.")
                A(f"> ⚠ **Provider Error:** {err_msg}\n")
                continue

            agents = result.get("agents", {})

            # ── TEAM 1: ANALYST TEAM ────────────────────────────── #
            A("#### 🔬 Team 1 — Analyst Team\n")

            # A1: Fundamental Analyst
            fund_a = agents.get("fundamental_analyst", {})
            if fund_a.get("findings") or fund_a.get("verdict") or fund_a.get("short_term_verdict"):
                A("##### A1 — Fundamental Analyst\n")
                for k, label in [("business_health","Business Health"),("valuation_stance","Valuation"),("capital_allocation","Capital Alloc"),("analyst_conviction","Analyst Conv.")]:
                    v = fund_a.get(k)
                    if v and v != "N/A": A(f"**{label}:** {v}  ")
                A("")
                st_v = fund_a.get("short_term_verdict","N/A"); lt_v = fund_a.get("long_term_verdict","N/A")
                A(f"**Short-term:** {st_v} — _{fund_a.get('short_term_rationale','')}_ ")
                A(f"**Long-term:** {lt_v} — _{fund_a.get('long_term_rationale','')}_\n")
                ks = fund_a.get("key_strengths",[]); kc = fund_a.get("key_concerns",[])
                if ks: A("**Strengths:** " + " | ".join(ks[:3]) + "  ")
                if kc: A("**Concerns:** " + " | ".join(kc[:3]) + "  ")
                for f in fund_a.get("findings", []): A(f"- {f}")
                if fund_a.get("verdict"): A(f"\n> *{fund_a['verdict']}*\n")
                A("")

            # A2: Sentiment Analyst
            sent_a = agents.get("sentiment_analyst", {})
            if sent_a.get("verdict") or sent_a.get("sentiment_bias"):
                A("##### A2 — Sentiment Analyst\n")
                A(f"**Bias:** {sent_a.get('sentiment_bias','N/A')}  ")
                A(f"**Mood:** {sent_a.get('market_mood','N/A')}  **Score:** {sent_a.get('sentiment_score',0):+.2f}  ")
                if sent_a.get("earnings_risk_flag"): A("**⚠ Earnings Risk Flag: YES**  ")
                A("")
                if sent_a.get("verdict"): A(f"> *{sent_a['verdict']}*\n")
                A("")

            # A3: News Analyst
            news_a = agents.get("news_analyst", {})
            if news_a.get("verdict") or news_a.get("vix_regime"):
                A("##### A3 — News Analyst\n")
                A(f"**VIX Regime:** {news_a.get('vix_regime','N/A')}  **Macro Bias:** {news_a.get('macro_bias','N/A')}  ")
                A(f"**Timing Caution:** {news_a.get('timing_caution','N/A')}  **Sector Timing:** {news_a.get('sector_timing','N/A')}  ")
                if news_a.get("earnings_risk"): A("**⚠ Earnings Risk: YES**  ")
                if news_a.get("override_short_term"): A(f"**Override ST:** YES — {news_a.get('override_rationale','')}  ")
                A("")
                if news_a.get("verdict"): A(f"> *{news_a['verdict']}*\n")
                A("")

            # A4: Technical Analyst
            tech_a = agents.get("technical_analyst", {})
            if tech_a.get("verdict") or tech_a.get("statistical_bias"):
                A("##### A4 — Technical Analyst\n")
                A(f"**Bias:** {tech_a.get('statistical_bias','N/A')}  **Worth:** {tech_a.get('worth_investing','N/A')}  ")
                d2 = tech_a.get("direction",""); e2 = tech_a.get("entry_price"); s2 = tech_a.get("stop_loss")
                if d2 and e2: A(f"**Trade:** {d2} Entry=${e2} Stop=${s2}  ")
                A("")
                st_p = tech_a.get("price_predictions",{}).get("short_term",{})
                lt_p = tech_a.get("price_predictions",{}).get("long_term",{})
                if any(st_p.get(k,{}).get("price") for k in ["1_week","2_weeks","3_weeks","1_month"]):
                    A("**Short-Term Predictions:**\n")
                    A("| Horizon | Price | Change% | Method |")
                    A("|---|---|---|---|")
                    for k, lbl in [("1_week","1 Week"),("2_weeks","2 Weeks"),("3_weeks","3 Weeks"),("1_month","1 Month")]:
                        d = st_p.get(k,{})
                        if d.get("price"):
                            chg = float(d.get("change_pct") or 0); sgn = "+" if chg >= 0 else ""
                            A(f"| {lbl} | ${d['price']} | {sgn}{chg:.1f}% | {d.get('method','')} |")
                    A("")
                if any(lt_p.get(k,{}).get("price") for k in ["3_months","6_months","9_months","12_months"]):
                    A("**Long-Term Predictions:**\n")
                    A("| Horizon | Price | Change% | Method |")
                    A("|---|---|---|---|")
                    for k, lbl in [("3_months","3 Months"),("6_months","6 Months"),("9_months","9 Months"),("12_months","12 Months")]:
                        d = lt_p.get(k,{})
                        if d.get("price"):
                            chg = float(d.get("change_pct") or 0); sgn = "+" if chg >= 0 else ""
                            A(f"| {lbl} | ${d['price']} | {sgn}{chg:.1f}% | {d.get('method','')} |")
                    A("")
                if tech_a.get("verdict"): A(f"> *{tech_a['verdict']}*\n")
                A("")

            # ── TEAM 2: RESEARCHER TEAM ─────────────────────────── #
            A("#### 🧠 Team 2 — Researcher Team\n")

            bull_r = agents.get("bullish_researcher",{})
            if bull_r.get("verdict") or bull_r.get("bull_stance"):
                A("##### R1 — Bullish Researcher\n")
                A(f"**Stance:** {bull_r.get('bull_stance','N/A')} (conf: {bull_r.get('bull_confidence','N/A')})  ")
                if bull_r.get("bull_thesis"): A(f"\n> {bull_r['bull_thesis']}\n")
                cats = bull_r.get("growth_catalysts",[])
                if cats: A("**Catalysts:** " + " | ".join(cats[:3]) + "  ")
                A("")

            bear_r = agents.get("bearish_researcher",{})
            if bear_r.get("verdict") or bear_r.get("bear_stance"):
                A("##### R2 — Bearish Researcher\n")
                A(f"**Stance:** {bear_r.get('bear_stance','N/A')} (conf: {bear_r.get('bear_confidence','N/A')})  ")
                if bear_r.get("bear_thesis"): A(f"\n> {bear_r['bear_thesis']}\n")
                risks_r = bear_r.get("key_risks",[])
                if risks_r: A("**Key Risks:** " + " | ".join(risks_r[:3]) + "  ")
                A("")

            synth = agents.get("synthesizer",{})
            if synth.get("verdict") or synth.get("net_bias"):
                A("##### R3 — Research Synthesizer\n")
                A(f"**Net Bias:** {synth.get('net_bias','N/A')} (strength: {synth.get('consensus_strength','N/A')})  ")
                if synth.get("balanced_brief"): A(f"\n> {synth['balanced_brief']}\n")
                agreed = synth.get("agreed_points",[])
                if agreed: A("**Agreed Points:** " + " | ".join(agreed[:2]) + "  ")
                A("")

            # ── TEAM 3: TRADING TEAM ─────────────────────────────── #
            A("#### 📈 Team 3 — Trading Team\n")

            for a_key, a_label, tgt_keys in [
                ("momentum_trader","T1 — Momentum Trader",[("target_1w","1W"),("target_2w","2W"),("target_3w","3W"),("target_1m","1M")]),
                ("value_trader",   "T2 — Value Trader",   [("target_3m","3M"),("target_6m","6M"),("target_9m","9M"),("target_12m","12M")]),
                ("swing_trader",   "T3 — Swing Trader",   [("target_1w","1W"),("target_2w","2W"),("target_3w","3W"),("target_1m","1M")]),
            ]:
                trader = agents.get(a_key,{})
                if trader.get("verdict") or trader.get("trade_action"):
                    A(f"##### {a_label}\n")
                    A(f"**Action:** {trader.get('trade_action','N/A')}  ")
                    e_t = trader.get("entry_price"); s_t = trader.get("stop_loss"); sz_t = trader.get("position_size_pct")
                    if e_t: A(f"**Entry:** ${e_t}  **Stop:** ${s_t}  **Size:** {sz_t}%  ")
                    tgt_parts = [f"{lbl}=${trader.get(k)}" for k,lbl in tgt_keys if trader.get(k)]
                    if tgt_parts: A("**Targets:** " + " | ".join(tgt_parts) + "  ")
                    if a_key == "value_trader":
                        mos = trader.get("margin_of_safety_pct")
                        vs = trader.get("value_score","N/A")
                        if mos: A(f"**MoS:** {mos:.1f}%  ")
                        A(f"**Value Score:** {vs}  ")
                    if a_key == "swing_trader" and trader.get("swing_setup"): A(f"**Setup:** {trader['swing_setup']}  ")
                    A("")
                    if trader.get("verdict"): A(f"> *{trader['verdict']}*\n")
                    A("")

            # ── TEAM 4: RISK MANAGEMENT ──────────────────────────── #
            A("#### 🛡️ Team 4 — Risk Management\n")

            mr = agents.get("market_risk",{})
            if mr.get("verdict") or mr.get("market_risk_level"):
                A("##### RM1 — Market Risk Agent\n")
                A(f"**Risk Level:** {mr.get('market_risk_level','N/A')}  **VIX Flag:** {mr.get('vix_risk_flag','N/A')}  ")
                A(f"**Timing:** {mr.get('timing_recommendation','N/A')}  **Max Position:** {mr.get('max_position_flag','N/A')}  ")
                A("")
                if mr.get("verdict"): A(f"> *{mr['verdict']}*\n")
                A("")

            pr = agents.get("portfolio_risk",{})
            if pr.get("verdict") or pr.get("recommended_size_pct") is not None:
                A("##### RM2 — Portfolio Risk Agent\n")
                sz_pr = pr.get("recommended_size_pct"); sc_pr = pr.get("scaling_factor","N/A")
                if sz_pr is not None: A(f"**Position Size:** {sz_pr}% of account (scale: {sc_pr})  ")
                ve = pr.get("validated_entry"); vs2 = pr.get("validated_stop"); sv2 = pr.get("stop_validation","N/A")
                if ve: A(f"**Validated Entry:** ${ve}  **Validated Stop:** ${vs2} ({sv2})  ")
                kelly = pr.get("kelly_fraction")
                if kelly is not None: A(f"**Kelly Fraction:** {kelly:.3f}  ")
                A("")
                if pr.get("verdict"): A(f"> *{pr['verdict']}*\n")
                A("")

            # ── THE JUDGE ─────────────────────────────────────────── #
            rec  = result.get("recommendation", "N/A")
            conf = result.get("confidence", "N/A")
            ost  = result.get("overall_short_term", "N/A")
            olt  = result.get("overall_long_term",  "N/A")
            A("#### ⚖️ The Judge — Overall Summary\n")
            A(f"**Recommendation: {rec}** _(Confidence: {conf})_\n")
            A(f"**Overall Short-term (1-4 wk):** {ost}  ")
            A(f"**Overall Long-term (3-12 mo):** {olt}\n")
            A(f"> {result.get('summary','')}\n")

            # Timing note
            timing_note = result.get("timing_note", "")
            if timing_note:
                A(f"> ⏱ **Timing Guidance:** {timing_note}\n")

            ep, ex, sl = result.get("entry_price"), result.get("exit_price"), result.get("stop_loss")
            direction  = result.get("trade_direction") or ""
            size_pct_r = result.get("position_size_pct")
            if ep or ex or sl:
                dir_note = f" ({direction})" if direction else ""
                A(f"##### Trading Levels{dir_note}\n")
                A("| Level | Price |")
                A("|---|---|")
                if ep: A(f"| Entry     | ${ep} |")
                if ex: A(f"| Exit      | ${ex} |")
                if sl: A(f"| Stop Loss | ${sl} |")
                if size_pct_r: A(f"| Size | {size_pct_r}% |")
                A("")

            tgt = result.get("target_prices", {})
            if tgt:
                A("##### AI Price Targets\n")
                A("| Horizon | Target Price | Accuracy |")
                A("|---|---|---|")
                label_map_r = {
                    "1_week": "1 Week", "2_weeks": "2 Weeks", "3_weeks": "3 Weeks",
                    "1_month": "1 Month", "3_months": "3 Months", "6_months": "6 Months",
                    "9_months": "9 Months", "12_months": "12 Months",
                }
                for key, label in label_map_r.items():
                    entry_data = tgt.get(key)
                    if isinstance(entry_data, dict):
                        price_val = entry_data.get("price")
                        acc_val   = entry_data.get("accuracy_pct")
                    elif isinstance(entry_data, (int, float)) and entry_data:
                        price_val = entry_data; acc_val = None
                    else:
                        price_val = None; acc_val = None
                    price_str = f"${price_val}" if price_val else "—"
                    acc_str   = f"{acc_val:.1f}%" if isinstance(acc_val, (int, float)) else "—"
                    A(f"| {label} | {price_str} | {acc_str} |")
                A("")

            A(f"**Technical Verdict:**   {result.get('technical_verdict','')}\n")
            A(f"**Fundamental Verdict:** {result.get('fundamental_verdict','')}\n")
            A(f"**Valuation Verdict:**   {result.get('valuation_verdict','')}\n")
            A(f"**Sentiment Verdict:**   {result.get('sentiment_verdict','')}\n")
            A(f"**Macro Verdict:**       {result.get('macro_verdict','')}\n")

            if result.get("key_bull_case"):
                A("##### Bull Case\n")
                for b in result["key_bull_case"]: A(f"- ✅ {b}")
                A("")
            if result.get("key_bear_case"):
                A("##### Bear Case\n")
                for b in result["key_bear_case"]: A(f"- ❌ {b}")
                A("")
            if result.get("key_risks"):
                A("##### Key Risks\n")
                for r in result["key_risks"]: A(f"- ⚠️  {r}")
                A("")
            if result.get("catalysts"):
                A("##### Catalysts\n")
                for c in result["catalysts"]: A(f"- ⭐ {c}")
                A("")

            # Recent news with dates
            news_dated = result.get("news_with_dates", [])
            if news_dated:
                A("##### Recent News & Market Factors\n")
                A("| Date | Headline | Publisher | Impact |")
                A("|---|---|---|---|")
                for n in news_dated:
                    if not isinstance(n, dict):
                        continue
                    imp  = n.get("impact", "NEUTRAL")
                    note = n.get("impact_note", "")
                    note_str = f" _{note}_" if note else ""
                    A(f"| {n.get('date','—')} | {n.get('title','')} | {n.get('publisher','—')} | **{imp}**{note_str} |")
                A("")

            # Team summaries
            team_summaries = result.get("team_summaries", {})
            if team_summaries:
                A("##### Team Summaries\n")
                for t_key, t_label in [
                    ("analyst_team",    "🔬 Analyst Team"),
                    ("researcher_team", "🔍 Researcher Team"),
                    ("trading_team",    "📈 Trading Team"),
                    ("risk_team",       "🛡 Risk Team"),
                ]:
                    ts = team_summaries.get(t_key, "")
                    if ts:
                        A(f"**{t_label}:** {ts}  ")
                A("")
            if result.get("alternative_pick"):
                A(f"**Alternative Pick:** `{result['alternative_pick']}` — {result.get('alternative_reason','')}\n")
            tu = result.get("token_usage", {})
            if tu.get("total_tokens"):
                A(f"*Tokens — Prompt: {tu['prompt_tokens']:,} · Completion: {tu['completion_tokens']:,} · Total: {tu['total_tokens']:,}*\n")

            # ── Raw Judge JSON ───────────────────────────────────── #
            _JUDGE_KEYS = [
                "recommendation", "confidence", "overall_short_term", "overall_long_term",
                "target_prices", "trade_direction", "entry_price", "exit_price", "stop_loss",
                "position_size_pct", "key_bull_case", "key_bear_case", "key_risks",
                "catalysts", "summary", "technical_verdict", "fundamental_verdict",
                "valuation_verdict", "sentiment_verdict", "macro_verdict",
                "alternative_pick", "alternative_reason",
            ]
            judge_json = {k: result[k] for k in _JUDGE_KEYS if k in result}
            if judge_json:
                A("<details>")
                A("<summary><b>Raw Judge JSON</b></summary>\n")
                A("```json")
                A(json.dumps(judge_json, indent=2))
                A("```")
                A("</details>\n")

            A("---\n")

            # ── placeholder closing (no old code below this line) ─── #
            if False:  # removed: mathematician, quant, market, shareholder blocks
                pass


    # Footer
    A("---")
    A(f"\n*Report generated by FinBot on {run_date}. "
      "This is not financial advice. All models are simplifications of reality.*\n")

    # ---- Save file ----------------------------------------------- #
    report_path = os.path.join(dir_path, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
