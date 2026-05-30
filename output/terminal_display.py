"""
output/terminal_display.py — Rich terminal output for FinBot.

Uses the `rich` library to display a beautifully formatted, colour-coded
analysis report directly in the terminal. Structured as a series of
panels and tables, one per analysis domain.
"""

from rich.console import Console, Group as RenderGroup
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.columns import Columns
import datetime
import math

console = Console()


# ------------------------------------------------------------------ #
# Colour helpers
# ------------------------------------------------------------------ #

def _colour_signal(signal: str) -> str:
    """Map a signal string to a Rich colour markup string."""
    signal_upper = signal.upper()
    if any(w in signal_upper for w in ("BUY", "UNDERVALUED", "BULLISH", "OVERSOLD", "UPWARD", "GOLDEN")):
        return f"[bold green]{signal}[/bold green]"
    if any(w in signal_upper for w in ("SELL", "OVERVALUED", "BEARISH", "OVERBOUGHT", "DOWNWARD", "DEATH")):
        return f"[bold red]{signal}[/bold red]"
    if any(w in signal_upper for w in ("HOLD", "NEUTRAL", "FAIR", "MODERATE", "N/A")):
        return f"[yellow]{signal}[/yellow]"
    if any(w in signal_upper for w in ("ERROR", "UNAVAILABLE")):
        return f"[dim]{signal}[/dim]"
    return signal


def _pct_colour(pct: float) -> str:
    """Colour a percentage change string (green positive, red negative)."""
    if pct > 0:
        return f"[green]+{pct:.2f}%[/green]"
    elif pct < 0:
        return f"[red]{pct:.2f}%[/red]"
    return f"{pct:.2f}%"


# ------------------------------------------------------------------ #
# Section 1: Header
# ------------------------------------------------------------------ #

def display_header(ticker: str, info: dict) -> None:
    name      = info.get("shortName", ticker)
    price     = info.get("currentPrice", "N/A")
    sector    = info.get("sector", "N/A")
    industry  = info.get("industry", "N/A")
    mcap      = info.get("marketCap")
    mcap_str  = f"${mcap:,.0f}" if mcap else "N/A"
    hi52      = info.get("fiftyTwoWeekHigh", "N/A")
    lo52      = info.get("fiftyTwoWeekLow",  "N/A")
    run_date  = datetime.date.today().isoformat()

    header_text = (
        f"[bold cyan]{name}[/bold cyan]   [bold white]({ticker})[/bold white]\n"
        f"[bold yellow]${price}[/bold yellow]  |  "
        f"52W High: [green]{hi52}[/green]  52W Low: [red]{lo52}[/red]\n"
        f"Sector: {sector}  |  Industry: {industry}\n"
        f"Market Cap: {mcap_str}  |  Report Date: {run_date}"
    )
    console.print()
    console.print(Panel(header_text, title="[bold]FinBot — AI Stock Analyzer[/bold]",
                        border_style="cyan", expand=False))


# ------------------------------------------------------------------ #
# Section 2: Basic Description
# ------------------------------------------------------------------ #

def display_basic_description(llm: dict, info: dict) -> None:
    """Render the company background panel (sourced from yfinance — no LLM)."""
    console.print(Rule("[bold cyan]COMPANY DESCRIPTION[/bold cyan]"))

    # Description is now always a plain yfinance dict at the top level
    desc = llm.get("description")
    if not desc:
        fallback = info.get("longBusinessSummary", "")
        if fallback:
            console.print(Panel(fallback, border_style="dim", expand=False))
        else:
            console.print("  [dim]No company description available.[/dim]")
        return

    name      = desc.get("company_name", info.get("shortName", ""))
    sector    = desc.get("sector", "N/A")
    industry  = desc.get("industry", "N/A")
    country   = desc.get("country", "N/A")
    website   = desc.get("website", "N/A")
    employees = desc.get("employees", "N/A")
    founded   = desc.get("founding_year", "Unknown")
    overview  = desc.get("business_overview", "")
    news      = desc.get("recent_news", [])

    meta_line = (
        f"[bold]{name}[/bold]  |  Sector: {sector}  |  Industry: {industry}\n"
        f"Country: {country}  |  Employees: {employees}  |  Est. ~{founded}"
        + (f"\n[dim]{website}[/dim]" if website and website != "N/A" else "")
    )
    body_lines = [meta_line]
    if overview:
        body_lines.append(f"\n{overview}")
    if news:
        body_lines.append("\n[bold]Recent News:[/bold]")
        for h in news[:5]:
            body_lines.append(f"  • {h}")

    console.print(Panel("\n".join(body_lines), border_style="cyan", expand=False))


# ------------------------------------------------------------------ #
# Section 3: Technical Analysis
# ------------------------------------------------------------------ #

def display_technical(technical: dict) -> None:
    console.print(Rule("[bold cyan]TECHNICAL ANALYSIS[/bold cyan]"))

    latest  = technical.get("latest", {})
    signals = technical.get("signals", {})
    fibs    = technical.get("fibonacci", {})
    pivots  = technical.get("pivot_levels", {})

    # ---- Indicator values table ------------------------------------ #
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
    tbl.add_column("Indicator",   style="cyan",  no_wrap=True)
    tbl.add_column("Value",       style="white", justify="right")
    tbl.add_column("Signal",      no_wrap=False)

    rows = [
        ("RSI (14)",           f"{latest.get('RSI', 'N/A')}",        signals.get("RSI", "")),
        ("MACD Line",          f"{latest.get('MACD_Line', 'N/A')}",  signals.get("MACD", "")),
        ("MACD Signal",        f"{latest.get('MACD_Signal', 'N/A')}", ""),
        ("MACD Histogram",     f"{latest.get('MACD_Hist', 'N/A')}",  ""),
        ("SMA-20",             f"{latest.get('SMA_20', 'N/A')}",     ""),
        ("SMA-50",             f"{latest.get('SMA_50', 'N/A')}",     signals.get("MA_Trend", "")),
        ("SMA-200",            f"{latest.get('SMA_200', 'N/A')}",    ""),
        ("EMA-12",             f"{latest.get('EMA_12', 'N/A')}",     ""),
        ("EMA-26",             f"{latest.get('EMA_26', 'N/A')}",     ""),
        ("Bollinger Upper",    f"{latest.get('BB_Upper', 'N/A')}",   signals.get("Bollinger", "")),
        ("Bollinger Middle",   f"{latest.get('BB_Middle', 'N/A')}",  ""),
        ("Bollinger Lower",    f"{latest.get('BB_Lower', 'N/A')}",   ""),
        ("BB %B",              f"{latest.get('BB_PctB', 'N/A')}",    ""),
    ]
    for indicator, value, signal in rows:
        tbl.add_row(indicator, value, _colour_signal(signal) if signal else "")

    console.print(tbl)

    # ---- Key Levels ------------------------------------------------ #
    lev_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", title="Key Price Levels")
    lev_tbl.add_column("Fibonacci Level", style="cyan")
    lev_tbl.add_column("Price",           justify="right")
    lev_tbl.add_column("",               style="dim", justify="left")

    for label, price in fibs.items():
        lev_tbl.add_row(label, f"${price}", "")

    console.print(lev_tbl)

    res = pivots.get("resistance_levels", [])
    sup = pivots.get("support_levels", [])
    console.print(f"  [red]Resistance pivots:[/red] {', '.join('$'+str(r) for r in res) or 'N/A'}")
    console.print(f"  [green]Support pivots:   [/green] {', '.join('$'+str(s) for s in sup) or 'N/A'}")


# ------------------------------------------------------------------ #
# Section 3: Fundamental Analysis
# ------------------------------------------------------------------ #

def display_fundamental(fundamental: dict) -> None:
    console.print(Rule("[bold cyan]FUNDAMENTAL ANALYSIS[/bold cyan]"))

    pe     = fundamental.get("pe", {})
    peg    = fundamental.get("peg", {})
    graham = fundamental.get("graham", {})
    dcf    = fundamental.get("dcf", {})
    pb_div = fundamental.get("pb_div", {})

    tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta")
    tbl.add_column("Metric",  style="cyan", no_wrap=True)
    tbl.add_column("Value",   justify="right")
    tbl.add_column("Signal / Interpretation")

    tbl.add_row("Trailing P/E",      str(pe.get("trailing_pe",  "N/A")), _colour_signal(pe.get("trailing_signal", "")))
    tbl.add_row("Forward P/E",       str(pe.get("forward_pe",   "N/A")), _colour_signal(pe.get("forward_signal",  "")))
    tbl.add_row("PEG Ratio",         str(peg.get("peg",         "N/A")), _colour_signal(peg.get("signal", "")))
    tbl.add_row("Graham Number",     f"${graham.get('graham_number', 'N/A')}", _colour_signal(graham.get("signal", "")))
    tbl.add_row("DCF Intrinsic",     f"${dcf.get('dcf_value', 'N/A')}",
                _colour_signal(dcf.get("signal", "")) + f"\n  [dim]Growth: {dcf.get('growth_rate_used','N/A')}  WACC: {dcf.get('wacc_used','N/A')}[/dim]")
    tbl.add_row("Price-to-Book",     str(pb_div.get("price_to_book", "N/A")), _colour_signal(pb_div.get("pb_signal", "")))
    tbl.add_row("Dividend Yield",    f"{pb_div.get('dividend_yield', 'N/A')}%", _colour_signal(pb_div.get("dividend_signal", "")))

    console.print(tbl)


# ------------------------------------------------------------------ #
# Section 4: Statistical Analysis
# ------------------------------------------------------------------ #

def display_statistical(statistical: dict) -> None:
    console.print(Rule("[bold cyan]STATISTICAL ANALYSIS[/bold cyan]"))

    vol     = statistical.get("volatility", {})
    beta    = statistical.get("beta", {})
    sharpe  = statistical.get("sharpe", {})
    sortino = statistical.get("sortino", {})
    drawdown= statistical.get("drawdown", {})
    reg     = statistical.get("regression", {})
    mc      = statistical.get("monte_carlo", {})

    # Risk metrics table
    risk_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", title="Risk Metrics")
    risk_tbl.add_column("Metric",  style="cyan")
    risk_tbl.add_column("Value",   justify="right")
    risk_tbl.add_column("Signal")

    risk_tbl.add_row(
        "Annual Volatility (σ)",
        f"{vol.get('sigma_annual', 'N/A') * 100:.2f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A",
        vol.get("label", "")
    )
    risk_tbl.add_row(
        "Beta (vs S&P 500)",
        str(beta.get("beta", "N/A")),
        _colour_signal(beta.get("label", ""))
    )
    risk_tbl.add_row(
        "Sharpe Ratio",
        str(sharpe.get("sharpe", "N/A")),
        _colour_signal(sharpe.get("signal", ""))
    )
    risk_tbl.add_row(
        "Annual Return (hist.)",
        sharpe.get("annual_return", "N/A"),
        ""
    )
    risk_tbl.add_row(
        "Sortino Ratio",
        str(sortino.get("sortino", "N/A")),
        _colour_signal(sortino.get("signal", ""))
    )
    risk_tbl.add_row(
        "Max Drawdown",
        f"{drawdown.get('max_drawdown_pct','N/A')}%" if drawdown.get('max_drawdown_pct') is not None else "N/A",
        ""
    )
    risk_tbl.add_row(
        "Calmar Ratio",
        str(drawdown.get("calmar", "N/A")),
        _colour_signal(drawdown.get("signal", ""))
    )
    console.print(risk_tbl)

    # Linear regression predictions
    reg_preds = reg.get("predictions", {})
    if reg_preds:
        reg_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta",
                        title=f"Linear Regression Predictions  [dim](R² = {reg.get('r_squared','N/A')}  Trend: {reg.get('trend_dir','')})[/dim]")
        reg_tbl.add_column("Horizon", style="cyan")
        reg_tbl.add_column("Price Target",  justify="right")
        reg_tbl.add_column("Expected Move", justify="right")
        for horizon, data in reg_preds.items():
            reg_tbl.add_row(horizon, f"${data['price']}", _pct_colour(data["change%"]))
        console.print(reg_tbl)

    # Monte Carlo predictions
    mc_preds = mc.get("predictions", {})
    if mc_preds:
        mc_params = mc.get('params', {})
        mc_mu_method = mc_params.get('mu_method', 'historical')
        mc_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta",
                       title=f"Monte Carlo GBM  [dim]({mc_params.get('n_paths',1000):,} paths | drift: {mc_mu_method})[/dim]")
        mc_tbl.add_column("Horizon",       style="cyan")
        mc_tbl.add_column("Bear (P10)",    justify="right", style="red")
        mc_tbl.add_column("Median (P50)",  justify="right", style="yellow")
        mc_tbl.add_column("Bull (P90)",    justify="right", style="green")
        for horizon, data in mc_preds.items():
            mc_tbl.add_row(
                horizon,
                f"${data['p10']} ({data['p10_chg%']:+.1f}%)",
                f"${data['median']} ({data['med_chg%']:+.1f}%)",
                f"${data['p90']} ({data['p90_chg%']:+.1f}%)",
            )
        console.print(mc_tbl)


# ------------------------------------------------------------------ #
# Section 5: Analyst Consensus
# ------------------------------------------------------------------ #

def display_analyst(analyst_data: dict) -> None:
    console.print(Rule("[bold cyan]ANALYST CONSENSUS[/bold cyan]"))

    rec_key = analyst_data.get("recommendation_key", "N/A").upper()
    pt      = analyst_data.get("price_target") or {}
    news    = analyst_data.get("news", [])
    recs    = analyst_data.get("recommendations", [])

    console.print(f"  Consensus:    {_colour_signal(rec_key)}")
    if pt:
        console.print(f"  Price Targets: Mean=[yellow]${pt.get('mean','N/A')}[/yellow]  "
                      f"Low=[red]${pt.get('low','N/A')}[/red]  "
                      f"High=[green]${pt.get('high','N/A')}[/green]")

    # ---- Analyst upgrade/downgrade history table ------------------- #
    if recs:
        console.print()
        recs_tbl = Table(
            box=box.SIMPLE_HEAD, header_style="bold magenta",
            title="Recent Analyst Actions (most recent first)"
        )
        recs_tbl.add_column("Date",       style="cyan",  no_wrap=True, min_width=10)
        recs_tbl.add_column("Firm",       style="white", min_width=20)
        recs_tbl.add_column("Action",     justify="center", min_width=11)
        recs_tbl.add_column("From",       justify="center", style="dim",   min_width=12)
        recs_tbl.add_column("To Grade",   justify="center", min_width=12)

        for r in recs[:15]:
            action   = r.get("action", "")
            to_grade = r.get("to_grade", "") or "N/A"
            fr_grade = r.get("from_grade", "") or "—"

            if "Upgrade" in action or "Initiation" in action:
                action_fmt = f"[bold green]{action}[/bold green]"
                to_fmt     = f"[green]{to_grade}[/green]"
            elif "Downgrade" in action:
                action_fmt = f"[bold red]{action}[/bold red]"
                to_fmt     = f"[red]{to_grade}[/red]"
            else:
                action_fmt = f"[yellow]{action}[/yellow]"
                to_fmt     = to_grade

            recs_tbl.add_row(
                r.get("date", ""),
                r.get("firm", ""),
                action_fmt,
                fr_grade,
                to_fmt,
            )
        console.print(recs_tbl)

        # Quick sentiment summary
        n_up   = sum(1 for r in recs[:10] if "Upgrade"   in r.get("action", ""))
        n_down = sum(1 for r in recs[:10] if "Downgrade" in r.get("action", ""))
        n_init = sum(1 for r in recs[:10] if "Initiation" in r.get("action", ""))
        if n_up > n_down:
            sentiment = "[bold green]Net Positive[/bold green]"
        elif n_down > n_up:
            sentiment = "[bold red]Net Negative[/bold red]"
        else:
            sentiment = "[yellow]Mixed / Neutral[/yellow]"
        console.print(
            f"  [dim]Last 10 actions:[/dim]  "
            f"[green]{n_up} upgrade(s)[/green]  "
            f"[red]{n_down} downgrade(s)[/red]  "
            f"[cyan]{n_init} new initiation(s)[/cyan]  →  Sentiment: {sentiment}"
        )
        console.print()

        # ---- Legend ------------------------------------------------ #
        legend_tbl = Table(
            box=box.SIMPLE_HEAD, header_style="bold white",
            title="[dim]Legend[/dim]", show_header=True, padding=(0, 1)
        )
        legend_tbl.add_column("Term",        style="bold cyan",  no_wrap=True, min_width=18)
        legend_tbl.add_column("Meaning",     style="white",      min_width=52)

        legend_rows = [
            # Actions
            ("[bold]── ACTIONS ──[/bold]",  ""),
            ("[green]Upgrade[/green]",       "Analyst raises the stock's rating (e.g. Hold → Buy). Bullish signal."),
            ("[red]Downgrade[/red]",         "Analyst cuts the stock's rating (e.g. Buy → Neutral). Bearish signal."),
            ("[cyan]Initiation[/cyan]",      "First-ever coverage by this firm — no prior grade exists."),
            ("[yellow]Maintained[/yellow]",  "Analyst reaffirms the existing rating without change."),
            ("[yellow]Reiterated[/yellow]",  "Same as Maintained — a formal restatement of the prior rating."),
            # Grade scale
            ("", ""),
            ("[bold]── GRADES ──[/bold]",    ""),
            ("[green]Strong Buy[/green]",     "Highest conviction bullish call. Also: Conviction Buy, Top Pick."),
            ("[green]Buy / Overweight[/green]","Bullish. Analyst expects the stock to outperform. Also: Outperform, Positive."),
            ("[yellow]Hold / Neutral / Market Perform[/yellow]",
                                             "Neutral — analyst expects the stock to return roughly in line with the broader market."),
            ("[yellow]Equal Weight[/yellow]",  "Same as Hold/Neutral. Used by Morgan Stanley, Barclays and others."),
            ("[yellow]Sector Perform[/yellow]", "The stock is expected to perform IN LINE with its sector peers — neither outperform "
                                             "nor underperform the sector average. Used by RBC, CIBC, and others. "
                                             "Equivalent to Hold / Neutral / Market Perform."),
            ("[red]Underperform[/red]",       "Mild bearish — expected to lag the market. Also: Underweight, Sector Underperform."),
            ("[red]Sell / Strong Sell[/red]", "Bearish — analyst expects material price decline."),
            # Column meanings
            ("", ""),
            ("[bold]── COLUMNS ──[/bold]",   ""),
            ("From",                         "Rating held by the analyst BEFORE today's action."),
            ("To Grade",                      "New rating AFTER today's action. This is the current live rating."),
        ]
        for term, meaning in legend_rows:
            legend_tbl.add_row(term, meaning)
        console.print(legend_tbl)

    if news:
        console.print("\n  [bold]Recent Headlines:[/bold]")
        for item in news[:5]:
            console.print(f"  • {item['title']} [dim]({item['publisher']})[/dim]")
    console.print()


# ------------------------------------------------------------------ #
# Section 6: LLM Recommendation
# ------------------------------------------------------------------ #

def _display_agent_panel(title: str, colour: str, agent: dict, extra_rows: list = None) -> None:
    """Render a single agent's findings as a compact panel."""
    lines = []
    renderable_extras = []  # Rich renderables (e.g. Table) that cannot go inside a string body
    error = agent.get("_error") or agent.get("_revision_error")
    if error:
        lines.append(f"  [bold red]Agent error:[/bold red] [dim red]{error}[/dim red]")
    for f in agent.get("findings", []):
        lines.append(f"  [dim]•[/dim] {f}")
    verdict = agent.get("verdict", "")
    if verdict and verdict != "Unavailable":
        lines.append(f"\n  [italic]{verdict}[/italic]")
    if extra_rows:
        for label, val in extra_rows:
            if isinstance(val, str):
                lines.append(f"  [bold]{label}:[/bold] {val}")
            else:
                # Rich renderable (Table, etc.) — collect for printing after the panel
                renderable_extras.append((label, val))
    body = "\n".join(lines) if lines else "[dim]No findings available.[/dim]"
    console.print(Panel(body, title=f"[bold {colour}]{title}[/bold {colour}]",
                        border_style=colour, expand=True))
    for label, renderable in renderable_extras:
        console.print(f"  [bold {colour}]{label}:[/bold {colour}]")
        console.print(renderable)


def _display_provider_result(result: dict) -> None:
    """Display the full 5-team, 13-agent result for a single provider."""
    if not result.get("llm_available", True) or result.get("recommendation") == "ERROR":
        err_msg = result.get("summary", "Unknown error.")
        console.print(Panel(
            f"[red]{err_msg}[/red]",
            title="[bold red]Provider Error[/bold red]",
            border_style="red",
        ))
        return

    agents = result.get("agents", {})

    # helper: verdict colour
    def _vc2(v):
        return "green" if v == "WORTH_INVESTING" else "red" if v == "NOT_WORTH_INVESTING" else "yellow"

    # ══════════════════════════════════════════════════════════════ #
    #  TEAM 1: ANALYST TEAM  (4 parallel agents)
    # ══════════════════════════════════════════════════════════════ #
    console.print(Rule("[bold cyan]── TEAM 1: ANALYST TEAM ──[/bold cyan]"))

    # ── A1 — Fundamental Analyst ───────────────────────────────── #
    fund_a = agents.get("fundamental_analyst", {})
    if fund_a:
        f_extras = []
        for k, lbl in [("business_health","Health"), ("valuation_stance","Valuation"),
                       ("capital_allocation","Capital Alloc"), ("analyst_conviction","Analyst Conv.")]:
            v = fund_a.get(k, "N/A")
            if v and v != "N/A":
                col = ("green" if v in ("STRONG","UNDERVALUED","EXCELLENT","LEADER")
                       else "red" if v in ("WEAK","OVERVALUED","POOR") else "yellow")
                f_extras.append((lbl, f"[{col}]{v}[/{col}]"))
        for k, lbl in [("short_term_verdict","ST Verdict"), ("long_term_verdict","LT Verdict")]:
            v   = fund_a.get(k, "N/A")
            rat = fund_a.get(k.replace("_verdict","_rationale"), "")
            f_extras.append((lbl, f"[{_vc2(v)}]{v}[/{_vc2(v)}]  [dim]{rat}[/dim]"))
        ks = fund_a.get("key_strengths", [])
        kc = fund_a.get("key_concerns",  [])
        if ks: f_extras.append(("Strengths", " | ".join(f"[green]{s}[/green]" for s in ks[:2])))
        if kc: f_extras.append(("Concerns",  " | ".join(f"[red]{c}[/red]"   for c in kc[:2])))
        _display_agent_panel("A1 — FUNDAMENTAL ANALYST", "blue", fund_a, f_extras)

    # ── A2 — Sentiment Analyst ─────────────────────────────────── #
    sent_a = agents.get("sentiment_analyst", {})
    if sent_a:
        s_extras = []
        bias = sent_a.get("sentiment_bias", "N/A")
        bc   = "green" if bias == "BULLISH" else "red" if bias == "BEARISH" else "yellow"
        s_extras.append(("Bias", f"[{bc}]{bias}[/{bc}]"))
        s_extras.append(("Mood", sent_a.get("market_mood", "N/A")))
        pressure = sent_a.get("behavioural_pressure", "N/A")
        pc = "green" if pressure == "BUYING_PRESSURE" else "red" if pressure == "SELLING_PRESSURE" else "yellow"
        s_extras.append(("Pressure", f"[{pc}]{pressure}[/{pc}]"))
        score = sent_a.get("sentiment_score", 0) or 0
        sc2 = "green" if score > 0.1 else "red" if score < -0.1 else "yellow"
        s_extras.append(("Score", f"[{sc2}]{score:+.2f}[/{sc2}]"))
        if sent_a.get("earnings_risk_flag"):
            s_extras.append(("Earnings Risk", "[red]YES[/red]"))
        _display_agent_panel("A2 — SENTIMENT ANALYST", "cyan", sent_a, s_extras)

    # ── A3 — News Analyst ──────────────────────────────────────── #
    news_a = agents.get("news_analyst", {})
    if news_a:
        n_extras = []
        vix = news_a.get("vix_regime", "N/A")
        vc  = "green" if vix == "CALM" else "red" if vix == "HIGH_FEAR" else "yellow"
        n_extras.append(("VIX Regime", f"[{vc}]{vix}[/{vc}]"))
        mb  = news_a.get("macro_bias", "N/A")
        mbc = "green" if mb == "RISK_ON" else "red" if mb == "RISK_OFF" else "yellow"
        n_extras.append(("Macro Bias", f"[{mbc}]{mb}[/{mbc}]"))
        tc  = news_a.get("timing_caution", "N/A")
        tcc = "red" if tc == "MAJOR" else "yellow" if tc == "MINOR" else "green"
        n_extras.append(("Timing Caution", f"[{tcc}]{tc}[/{tcc}]"))
        n_extras.append(("Sector Timing", news_a.get("sector_timing", "N/A")))
        if news_a.get("earnings_risk"):
            n_extras.append(("Earnings Risk", "[red]YES[/red]"))
        if news_a.get("override_short_term"):
            n_extras.append(("Override ST", f"[red]YES — {news_a.get('override_rationale','')}[/red]"))
        _display_agent_panel("A3 — NEWS ANALYST", "yellow", news_a, n_extras)

    # ── A4 — Technical Analyst ─────────────────────────────────── #
    tech_a = agents.get("technical_analyst", {})
    if tech_a:
        t_extras = []
        bias = tech_a.get("statistical_bias", "N/A")
        bc   = "green" if bias == "BULLISH" else "red" if bias == "BEARISH" else "yellow"
        t_extras.append(("Bias", f"[{bc}]{bias}[/{bc}]"))
        direction = tech_a.get("direction", "")
        entry     = tech_a.get("entry_price")
        stop      = tech_a.get("stop_loss")
        if direction and entry:
            dc = "green" if direction == "LONG" else "red"
            t_extras.append(("Trade Setup",
                f"[{dc}]{direction}[/{dc}]  Entry=[green]${entry}[/green]  Stop=[red]${stop}[/red]"))
        worth = tech_a.get("worth_investing", "N/A")
        wc    = "green" if worth == "YES" else "red" if worth == "NO" else "yellow"
        t_extras.append(("Worth", f"[{wc}]{worth}[/{wc}]  [dim]{tech_a.get('worth_investing_rationale','')}[/dim]"))
        # Short-term price predictions table
        st_preds = tech_a.get("price_predictions", {}).get("short_term", {})
        lt_preds = tech_a.get("price_predictions", {}).get("long_term",  {})
        if any(st_preds.get(k, {}).get("price") for k in ["1_week","2_weeks","3_weeks","1_month"]):
            st_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold green",
                           title="Short-Term Predictions", show_edge=False, padding=(0,1))
            st_tbl.add_column("Horizon", style="cyan",   no_wrap=True)
            st_tbl.add_column("Price",   justify="right", no_wrap=True)
            st_tbl.add_column("Chg%",    justify="right", no_wrap=True)
            st_tbl.add_column("Method",  style="dim",     no_wrap=True)
            for k, lbl in [("1_week","1 Week"),("2_weeks","2 Weeks"),
                           ("3_weeks","3 Weeks"),("1_month","1 Month")]:
                d = st_preds.get(k, {})
                p = d.get("price")
                c = d.get("change_pct", 0) or 0
                if p:
                    st_tbl.add_row(lbl, f"[yellow]${p}[/yellow]", _pct_colour(c), d.get("method",""))
            t_extras.append(("Short-Term", st_tbl))
        if any(lt_preds.get(k, {}).get("price") for k in ["3_months","6_months","9_months","12_months"]):
            lt_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold blue",
                           title="Long-Term Predictions", show_edge=False, padding=(0,1))
            lt_tbl.add_column("Horizon", style="cyan",   no_wrap=True)
            lt_tbl.add_column("Price",   justify="right", no_wrap=True)
            lt_tbl.add_column("Chg%",    justify="right", no_wrap=True)
            lt_tbl.add_column("Method",  style="dim",     no_wrap=True)
            for k, lbl in [("3_months","3 Months"),("6_months","6 Months"),
                           ("9_months","9 Months"),("12_months","12 Months")]:
                d = lt_preds.get(k, {})
                p = d.get("price")
                c = d.get("change_pct", 0) or 0
                if p:
                    lt_tbl.add_row(lbl, f"[yellow]${p}[/yellow]", _pct_colour(c), d.get("method",""))
            t_extras.append(("Long-Term", lt_tbl))
        _display_agent_panel("A4 — TECHNICAL ANALYST", "magenta", tech_a, t_extras)

    # ══════════════════════════════════════════════════════════════ #
    #  TEAM 2: RESEARCHER TEAM  (Bull → Bear → Synthesizer)
    # ══════════════════════════════════════════════════════════════ #
    console.print(Rule("[bold green]── TEAM 2: RESEARCHER TEAM ──[/bold green]"))

    # ── R1 — Bullish Researcher ────────────────────────────────── #
    bull_r = agents.get("bullish_researcher", {})
    if bull_r:
        b_extras = []
        stance = bull_r.get("bull_stance", "N/A")
        sc = "green" if stance == "STRONG" else "yellow" if stance == "MODERATE" else "red"
        b_extras.append(("Stance", f"[{sc}]{stance}[/{sc}]  (conf: {bull_r.get('bull_confidence','N/A')})"))
        thesis = bull_r.get("bull_thesis", "")
        if thesis: b_extras.append(("Thesis", f"[dim]{thesis}[/dim]"))
        cats = bull_r.get("growth_catalysts", [])
        if cats: b_extras.append(("Catalysts", " | ".join(cats[:3])))
        _display_agent_panel("R1 — BULLISH RESEARCHER", "green", bull_r, b_extras)

    # ── R2 — Bearish Researcher ────────────────────────────────── #
    bear_r = agents.get("bearish_researcher", {})
    if bear_r:
        br_extras = []
        stance = bear_r.get("bear_stance", "N/A")
        sc = "red" if stance == "STRONG" else "yellow" if stance == "MODERATE" else "green"
        br_extras.append(("Stance", f"[{sc}]{stance}[/{sc}]  (conf: {bear_r.get('bear_confidence','N/A')})"))
        thesis = bear_r.get("bear_thesis", "")
        if thesis: br_extras.append(("Thesis", f"[dim]{thesis}[/dim]"))
        risks = bear_r.get("key_risks", [])
        if risks: br_extras.append(("Key Risks", " | ".join(risks[:3])))
        _display_agent_panel("R2 — BEARISH RESEARCHER", "red", bear_r, br_extras)

    # ── R3 — Research Synthesizer ──────────────────────────────── #
    synth = agents.get("synthesizer", {})
    if synth:
        sy_extras = []
        bias = synth.get("net_bias", "N/A")
        bc   = "green" if bias == "BULLISH" else "red" if bias == "BEARISH" else "yellow"
        sy_extras.append(("Net Bias",
            f"[{bc}]{bias}[/{bc}]  (strength: {synth.get('consensus_strength','N/A')})"))
        brief = synth.get("balanced_brief", "")
        if brief: sy_extras.append(("Brief", f"[dim]{brief}[/dim]"))
        agreed = synth.get("agreed_points", [])
        if agreed: sy_extras.append(("Agreed", " | ".join(agreed[:2])))
        _display_agent_panel("R3 — RESEARCH SYNTHESIZER", "yellow", synth, sy_extras)

    # ══════════════════════════════════════════════════════════════ #
    #  TEAM 3: TRADING TEAM  (3 parallel agents)
    # ══════════════════════════════════════════════════════════════ #
    console.print(Rule("[bold blue]── TEAM 3: TRADING TEAM ──[/bold blue]"))

    for agent_key, agent_label, colour, tgt_keys in [
        ("momentum_trader", "T1 — MOMENTUM TRADER", "cyan",
         [("target_1w","1W"), ("target_2w","2W"), ("target_3w","3W"), ("target_1m","1M")]),
        ("value_trader",    "T2 — VALUE TRADER",    "blue",
         [("target_3m","3M"), ("target_6m","6M"), ("target_9m","9M"), ("target_12m","12M")]),
        ("swing_trader",    "T3 — SWING TRADER",    "magenta",
         [("target_1w","1W"), ("target_2w","2W"), ("target_3w","3W"), ("target_1m","1M")]),
    ]:
        trader = agents.get(agent_key, {})
        if trader:
            tr_extras = []
            action = trader.get("trade_action", "N/A")
            ac = ("green"  if action in ("ENTER_LONG","HOLD") else
                  "red"    if action in ("ENTER_SHORT","EXIT") else "yellow")
            tr_extras.append(("Action", f"[{ac}]{action}[/{ac}]"))
            entry_t = trader.get("entry_price")
            stop_t  = trader.get("stop_loss")
            size_t  = trader.get("position_size_pct")
            if entry_t:
                stop_str = f"  Stop=[red]${stop_t}[/red]" if stop_t else ""
                size_str = f"  Size={size_t}%" if size_t else ""
                tr_extras.append(("Trade", f"Entry=[green]${entry_t}[/green]{stop_str}{size_str}"))
            tgt_parts = []
            for tgt_key, tgt_lbl in tgt_keys:
                v = trader.get(tgt_key)
                if v: tgt_parts.append(f"{tgt_lbl}=[yellow]${v}[/yellow]")
            if tgt_parts:
                tr_extras.append(("Targets", " | ".join(tgt_parts)))
            if agent_key == "value_trader":
                mos = trader.get("margin_of_safety_pct")
                vs  = trader.get("value_score", "N/A")
                if mos: tr_extras.append(("MoS", f"{mos:.1f}%"))
                tr_extras.append(("Value Score", vs))
            if agent_key == "swing_trader":
                tr_extras.append(("Setup", trader.get("swing_setup", "N/A")))
            _display_agent_panel(agent_label, colour, trader, tr_extras)

    # ══════════════════════════════════════════════════════════════ #
    #  TEAM 4: RISK MANAGEMENT  (2 parallel agents)
    # ══════════════════════════════════════════════════════════════ #
    console.print(Rule("[bold red]── TEAM 4: RISK MANAGEMENT ──[/bold red]"))

    # ── RM1 — Market Risk Agent ────────────────────────────────── #
    mr = agents.get("market_risk", {})
    if mr:
        mr_extras = []
        risk_lvl = mr.get("market_risk_level", "N/A")
        rc = "red" if risk_lvl in ("HIGH","EXTREME") else "yellow" if risk_lvl == "MODERATE" else "green"
        mr_extras.append(("Risk Level", f"[{rc}]{risk_lvl}[/{rc}]"))
        vix_flag = mr.get("vix_risk_flag", "N/A")
        vfc = "red" if vix_flag == "DANGER" else "yellow" if vix_flag == "CAUTION" else "green"
        mr_extras.append(("VIX Flag", f"[{vfc}]{vix_flag}[/{vfc}]"))
        timing = mr.get("timing_recommendation", "N/A")
        tc2 = "green" if timing == "PROCEED" else "red" if timing in ("WAIT","AVOID") else "yellow"
        mr_extras.append(("Timing", f"[{tc2}]{timing}[/{tc2}]"))
        mr_extras.append(("Max Position", mr.get("max_position_flag", "N/A")))
        _display_agent_panel("RM1 — MARKET RISK AGENT", "red", mr, mr_extras)

    # ── RM2 — Portfolio Risk Agent ─────────────────────────────── #
    pr = agents.get("portfolio_risk", {})
    if pr:
        pr_extras = []
        size_pr = pr.get("recommended_size_pct")
        method  = pr.get("sizing_method",  "N/A")
        scale   = pr.get("scaling_factor", "N/A")
        sc3 = "green" if scale == "FULL" else "red" if scale == "HALF" else "yellow"
        if size_pr is not None:
            pr_extras.append(("Position Size",
                f"[{sc3}]{size_pr}% of account[/{sc3}]  [dim](method: {method}, scale: {scale})[/dim]"))
        v_entry = pr.get("validated_entry")
        v_stop  = pr.get("validated_stop")
        sv      = pr.get("stop_validation", "N/A")
        stop_dist = pr.get("stop_distance_usd")
        if v_entry:
            pr_extras.append(("Validated Entry", f"[green]${v_entry}[/green]"))
        if v_stop:
            svc = "green" if sv == "VALID" else "red"
            pr_extras.append(("Validated Stop", f"[{svc}]${v_stop}[/{svc}]  [dim]({sv})[/dim]"))
        if stop_dist:
            pr_extras.append(("Stop Distance", f"${stop_dist:.4f}"))
        kelly = pr.get("kelly_fraction")
        if kelly is not None:
            pr_extras.append(("Kelly Fraction", f"{kelly:.3f}"))
        _display_agent_panel("RM2 — PORTFOLIO RISK AGENT", "yellow", pr, pr_extras)

    # ══════════════════════════════════════════════════════════════ #
    #  THE JUDGE — OVERALL SUMMARY
    # ══════════════════════════════════════════════════════════════ #
    console.print(Rule("[bold white]── THE JUDGE — OVERALL SUMMARY ──[/bold white]"))

    rec        = result.get("recommendation", "N/A")
    conf       = result.get("confidence", "N/A")
    summary    = result.get("summary", "")
    bull_cases = result.get("key_bull_case", [])
    bear_cases = result.get("key_bear_case", [])
    risks      = result.get("key_risks", [])
    catalysts  = result.get("catalysts", [])
    targets    = result.get("target_prices", {})
    entry      = result.get("entry_price")
    exit_p     = result.get("exit_price")
    stop       = result.get("stop_loss")
    direction  = result.get("trade_direction")
    size_pct   = result.get("position_size_pct")
    ost        = result.get("overall_short_term", "N/A")
    olt        = result.get("overall_long_term",  "N/A")
    timing_note   = result.get("timing_note", "")
    team_summaries= result.get("team_summaries", {})
    news_dated    = result.get("news_with_dates", [])

    # Recommendation badge
    rec_colour = "green" if "BUY" in rec else "red" if "SELL" in rec else "yellow"
    console.print(Panel(
        f"[bold {rec_colour}]{rec}[/bold {rec_colour}]\n[dim]Confidence: {conf}[/dim]",
        title="Recommendation", border_style=rec_colour, width=30,
    ))

    # Overall ST/LT verdicts
    def _vc(v):
        return "green" if v == "WORTH_INVESTING" else "red" if v == "NOT_WORTH_INVESTING" else "yellow"
    console.print(
        f"  [bold]Short-term:[/bold] [{_vc(ost)}]{ost}[/{_vc(ost)}]   "
        f"[bold]Long-term:[/bold]  [{_vc(olt)}]{olt}[/{_vc(olt)}]"
    )

    # Timing note
    if timing_note:
        console.print(Panel(
            f"[yellow]{timing_note}[/yellow]",
            title="[bold yellow]⏱  Timing Guidance[/bold yellow]",
            border_style="yellow",
        ))
    if entry or exit_p or stop:
        dir_colour = "green" if direction == "LONG" else "red" if direction == "SHORT" else "white"
        dir_label  = f" [{dir_colour}]({direction})[/{dir_colour}]" if direction else ""
        lvl_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta",
                        title=f"Trading Levels{dir_label}")
        lvl_tbl.add_column("Level", style="cyan")
        lvl_tbl.add_column("Price", justify="right")
        if entry:    lvl_tbl.add_row("Entry",     f"[green]${entry}[/green]")
        if exit_p:   lvl_tbl.add_row("Exit",      f"[cyan]${exit_p}[/cyan]")
        if stop:     lvl_tbl.add_row("Stop Loss", f"[red]${stop}[/red]")
        if size_pct: lvl_tbl.add_row("Size",      f"[yellow]{size_pct}%[/yellow]")
        console.print(lvl_tbl)

    # Price targets (new structure: {"price": float, "accuracy_pct": float})
    label_map = {
        "1_week": "1 Week", "2_weeks": "2 Weeks", "3_weeks": "3 Weeks",
        "1_month": "1 Month", "3_months": "3 Months", "6_months": "6 Months",
        "9_months": "9 Months", "12_months": "12 Months",
    }
    if targets:
        tgt_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", title="AI Price Targets")
        tgt_tbl.add_column("Horizon",  style="cyan")
        tgt_tbl.add_column("Target",   justify="right", style="yellow")
        tgt_tbl.add_column("Accuracy", justify="right", style="dim")
        for key, label in label_map.items():
            entry_data = targets.get(key)
            if isinstance(entry_data, dict):
                price_val = entry_data.get("price")
                acc_val   = entry_data.get("accuracy_pct")
            elif isinstance(entry_data, (int, float)) and entry_data:
                price_val = entry_data
                acc_val   = None
            else:
                price_val = None
                acc_val   = None
            price_str = f"${price_val}" if price_val else "—"
            acc_str   = f"{acc_val:.1f}%" if isinstance(acc_val, (int, float)) else "—"
            tgt_tbl.add_row(label, price_str, acc_str)
        console.print(tgt_tbl)

    # Summary & verdicts
    console.print(f"\n  [bold]Summary:[/bold] {summary}\n")
    for label, key in [
        ("Technical:  ", "technical_verdict"),
        ("Fundamental:", "fundamental_verdict"),
        ("Valuation:  ", "valuation_verdict"),
        ("Sentiment:  ", "sentiment_verdict"),
        ("Macro:      ", "macro_verdict"),
    ]:
        v = result.get(key, "")
        if v and v not in ("N/A", ""):
            console.print(f"  [bold]{label}[/bold] {v}")

    # Bull / Bear cases
    if bull_cases or bear_cases:
        console.print()
        if bull_cases:
            console.print("[bold green]Bull Case:[/bold green]")
            for b in bull_cases: console.print(f"  [green]✓[/green] {b}")
        if bear_cases:
            console.print("[bold red]Bear Case:[/bold red]")
            for b in bear_cases: console.print(f"  [red]✗[/red] {b}")

    if risks:
        console.print("\n[bold red]Key Risks:[/bold red]")
        for r in risks: console.print(f"  ⚠  {r}")

    if catalysts:
        console.print("[bold yellow]Catalysts:[/bold yellow]")
        for c in catalysts: console.print(f"  ★  {c}")

    # Recent news with dates
    if news_dated:
        console.print()
        console.print(Rule("[bold cyan]── Recent News & Market Factors ──[/bold cyan]"))
        news_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", show_header=True)
        news_tbl.add_column("Date",      style="dim",    no_wrap=True, width=12)
        news_tbl.add_column("Headline",  style="white",  ratio=4)
        news_tbl.add_column("Publisher", style="dim",    ratio=1)
        news_tbl.add_column("Impact",    justify="center", width=10)
        for n in news_dated:
            if not isinstance(n, dict):
                continue
            imp = n.get("impact", "NEUTRAL")
            imp_col = "green" if imp == "BULLISH" else "red" if imp == "BEARISH" else "yellow"
            imp_str = f"[{imp_col}]{imp}[/{imp_col}]"
            note = n.get("impact_note", "")
            headline_text = n.get("title", "")
            if note:
                headline_text = f"{headline_text}\n[dim]{note}[/dim]"
            news_tbl.add_row(
                n.get("date", "—"),
                headline_text,
                n.get("publisher", "—"),
                imp_str,
            )
        console.print(news_tbl)

    # Team summaries
    if team_summaries:
        console.print()
        console.print(Rule("[bold cyan]── Team Summaries ──[/bold cyan]"))
        for team_key, team_label in [
            ("analyst_team",    "🔬 Analyst Team"),
            ("researcher_team", "🔍 Researcher Team"),
            ("trading_team",    "📈 Trading Team"),
            ("risk_team",       "🛡  Risk Team"),
        ]:
            ts = team_summaries.get(team_key, "")
            if ts:
                console.print(f"  [bold cyan]{team_label}:[/bold cyan] {ts}")
        console.print()

    # Alternative pick
    alt = result.get("alternative_pick")
    if alt:
        console.print(
            f"\n  [bold yellow]Alternative Pick:[/bold yellow] [yellow]{alt}[/yellow]"
            f" — {result.get('alternative_reason','')}"
        )

    # Token usage
    tu = result.get("token_usage", {})
    if tu.get("total_tokens"):
        console.print(
            f"[dim]  Tokens — Prompt: {tu['prompt_tokens']:,}  "
            f"Completion: {tu['completion_tokens']:,}  "
            f"Total: {tu['total_tokens']:,}[/dim]"
        )

    console.print()


# ------------------------------------------------------------------ #
# Rule-based Judge display (--no-llm mode)
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# Rule-based Judge display (--no-llm mode)
# ------------------------------------------------------------------ #

def _render_price_outlook(rbj: dict, statistical: dict, info: dict) -> None:
    """Render a Price Outlook panel showing entry, target, and horizon forecasts."""
    current = info.get("currentPrice")
    if current is None:
        return

    mc_preds  = statistical.get("monte_carlo", {}).get("predictions", {})
    reg       = statistical.get("regression", {})
    slope     = reg.get("slope_daily")
    reg_preds = reg.get("predictions", {})

    # ── Tomorrow: OLS trend extrapolated 1 day ───────────────────── #
    if slope is not None:
        tomorrow     = current * math.exp(slope)
        tomorrow_chg = (tomorrow - current) / current * 100
    else:
        tomorrow     = None
        tomorrow_chg = None

    # ── Entry suggestion ─────────────────────────────────────────── #
    rec = rbj.get("recommendation", "")
    if rec == "STRONG BUY":
        entry_str = f"[bold green]${current:.2f}[/bold green]  [dim](market — strong entry)[/dim]"
    elif rec == "BUY":
        entry_str = f"[green]${current:.2f}[/green]  [dim](market — recommended entry)[/dim]"
    elif rec == "HOLD":
        entry_str = f"[yellow]${current:.2f}[/yellow]  [dim](hold existing; add only on pullback)[/dim]"
    else:
        entry_str = "[dim]N/A  (exit / avoid recommended)[/dim]"

    # ── 1-Year MC median as price target ─────────────────────────── #
    mc_1yr   = mc_preds.get("1 Year", {})
    mc_tgt   = mc_1yr.get("median")
    if mc_tgt is not None:
        tgt_chg = (mc_tgt - current) / current * 100
        tgt_col = "green" if tgt_chg >= 0 else "red"
        target_str = (
            f"[bold {tgt_col}]${mc_tgt:.2f}[/bold {tgt_col}]"
            f"  [dim]({tgt_chg:+.1f}%, 1-Year GBM median)[/dim]"
        )
    else:
        target_str = "[dim]N/A[/dim]"

    header = (
        f"  Current Price:   [bold yellow]${current:.2f}[/bold yellow]\n"
        f"  Suggested Entry: {entry_str}\n"
        f"  Price Target:    {target_str}\n"
    )

    # ── Forecast table ────────────────────────────────────────────── #
    tbl = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        padding=(0, 1),
        expand=False,
    )
    tbl.add_column("Horizon",         style="bold white", min_width=11)
    tbl.add_column("Trend (OLS)",     justify="right",    min_width=16)
    tbl.add_column("MC Bear (P10)",   justify="right",    min_width=15)
    tbl.add_column("MC Median",       justify="right",    min_width=18)
    tbl.add_column("MC Bull (P90)",   justify="right",    min_width=15)

    # Tomorrow row — trend only, no MC 1-day paths
    if tomorrow is not None:
        chg_col = "green" if tomorrow_chg >= 0 else "red"
        tbl.add_row(
            "Tomorrow†",
            f"[{chg_col}]${tomorrow:.2f}  ({tomorrow_chg:+.2f}%)[/{chg_col}]",
            "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
        )

    for label in ("1 Week", "1 Month", "3 Months", "6 Months", "9 Months", "1 Year"):
        mc  = mc_preds.get(label, {})
        rp  = reg_preds.get(label, {})

        reg_price = rp.get("price")
        reg_chg   = rp.get("change%")
        p10       = mc.get("p10")
        median    = mc.get("median")
        p90       = mc.get("p90")
        med_chg   = mc.get("med_chg%")

        reg_col = "green" if (reg_chg or 0) >= 0 else "red"
        med_col = "green" if (med_chg or 0) >= 0 else "red"

        reg_str = (
            f"[{reg_col}]${reg_price:.2f}  ({reg_chg:+.2f}%)[/{reg_col}]"
            if reg_price is not None else "[dim]—[/dim]"
        )
        p10_str = f"[red]${p10:.2f}[/red]"       if p10    else "[dim]—[/dim]"
        med_str = (
            f"[{med_col}]${median:.2f}  ({med_chg:+.1f}%)[/{med_col}]"
            if median is not None and med_chg is not None else "[dim]—[/dim]"
        )
        p90_str = f"[green]${p90:.2f}[/green]"   if p90    else "[dim]—[/dim]"

        tbl.add_row(label, reg_str, p10_str, med_str, p90_str)

    footer = (
        "\n  [dim]† Tomorrow = OLS linear regression trend (+1 trading day). "
        "All other horizons: Geometric Brownian Motion Monte Carlo (1,000 paths). "
        "P10/P90 represent the 10th/90th percentile of simulated outcomes.[/dim]"
    )

    console.print(Panel(
        RenderGroup(
            Text.from_markup(header),
            tbl,
            Text.from_markup(footer),
        ),
        title="[bold]PRICE OUTLOOK[/bold]",
        border_style="green",
    ))
    console.print()


def display_rule_based_judge(
    llm:        dict,
    statistical: dict | None = None,
    info:        dict | None = None,
    title:       str  = "RULE-BASED INVESTMENT JUDGE",
) -> None:
    """Render the deterministic four-lens judge results in the terminal."""
    rbj = llm.get("rule_based_judge")
    if not rbj:
        return

    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))

    composite = rbj.get("composite_score", 0)
    rec       = rbj.get("recommendation", "N/A")
    conf      = rbj.get("confidence", "N/A")
    worth     = rbj.get("worth_investing", "N/A")

    rec_colour = {
        "STRONG BUY":  "bold green",
        "BUY":         "green",
        "HOLD":        "yellow",
        "SELL":        "red",
        "STRONG SELL": "bold red",
    }.get(rec, "white")

    worth_colour = {
        "WORTH_INVESTING":     "green",
        "CONDITIONAL":        "yellow",
        "NOT_WORTH_INVESTING": "red",
    }.get(worth, "white")

    def _lens_panel(title: str, lens: dict, border_colour: str) -> Panel:
        lines = []
        lines.append(f"Score: [bold]{lens.get('score', 'N/A')}/100[/bold]  Grade: [bold]{lens.get('grade', '?')}[/bold]  Verdict: [bold {border_colour}]{lens.get('verdict', 'N/A')}[/bold {border_colour}]")
        lines.append("")
        for sig in lens.get("signals", []):
            pts     = sig.get("points", 0)
            possible = 2  # most signals are out of 2
            dot = "[green]●[/green]" if pts >= 2 else ("[yellow]●[/yellow]" if pts == 1 else "[red]●[/red]")
            lines.append(f"  {dot} {sig['metric']}: {sig.get('value','N/A')} — {sig.get('label','N/A')}")
        return Panel(
            "\n".join(lines),
            title=title,
            border_style=border_colour,
            expand=True,
        )

    fund_panel = _lens_panel(
        "[bold]FUNDAMENTAL LENS[/bold]",
        rbj.get("fundamental", {}),
        "blue",
    )
    tech_panel = _lens_panel(
        "[bold]TECHNICAL LENS[/bold]",
        rbj.get("technical", {}),
        "magenta",
    )
    val_panel  = _lens_panel(
        "[bold]VALUATION LENS[/bold]",
        rbj.get("valuation", {}),
        "cyan",
    )
    risk_panel = _lens_panel(
        "[bold]RISK LENS[/bold]",
        rbj.get("risk", {}),
        "red",
    )

    from rich.columns import Columns
    console.print(Columns([fund_panel, tech_panel, val_panel, risk_panel], equal=True, expand=True))
    console.print()

    veto_line = ""
    if rbj.get("veto_triggered"):
        veto_line = f"\n  [bold red]⚠ {rbj.get('veto_reason', 'Veto triggered')}[/bold red]"

    console.print(Panel(
        f"  Composite Score: [bold]{composite:.0f}/100[/bold]\n"
        f"  Recommendation:  [{rec_colour}]{rec}[/{rec_colour}]  (Confidence: {conf})\n"
        f"  Worth Investing: [{worth_colour}]{worth}[/{worth_colour}]{veto_line}\n\n"
        f"  {rbj.get('summary', '')}",
        title="[bold]OVERALL VERDICT[/bold]",
        border_style="cyan",
    ))
    console.print()

    if statistical and info:
        _render_price_outlook(rbj, statistical, info)


def display_llm(llm: dict, statistical: dict | None = None, info: dict | None = None) -> None:
    # ================================================================ #
    #  SECTION BANNER — makes it crystal-clear where LLM content lives  #
    # ================================================================ #
    console.print()
    console.print(Panel(
        "[bold cyan]The content below is generated by an AI language model.\n"
        "All price targets, verdicts, and narratives are LLM outputs — "
        "not guaranteed to be accurate.[/bold cyan]",
        title="[bold cyan]◈  AI ANALYST PIPELINE  ◈[/bold cyan]",
        border_style="cyan",
        expand=True,
    ))
    console.print()

    if not llm.get("llm_available"):
        console.print(Panel(f"[yellow]{llm.get('summary', 'LLM not available')}[/yellow]",
                            title="LLM Analysis Unavailable", border_style="yellow"))
        # No LLM ran — go straight to the deterministic judge
        console.print()
        console.print(Panel(
            "[bold white]The section below uses only deterministic, rule-based scoring.\n"
            "No AI model is involved. Results are fully reproducible.[/bold white]",
            title="[bold white]◈  DETERMINISTIC RULE-BASED JUDGE  ◈[/bold white]",
            border_style="white",
            expand=True,
        ))
        console.print()
        display_rule_based_judge(llm, statistical=statistical, info=info)
        return

    providers = llm.get("providers", {})
    if not providers:
        console.print(Panel("[yellow]No provider results available.[/yellow]",
                            title="LLM Analysis Unavailable", border_style="yellow"))
        return

    for provider_name, result in providers.items():
        console.print(Rule(f"[bold white]── {provider_name} ──[/bold white]"))
        _display_provider_result(result)

    # ================================================================ #
    #  SECTION BANNER — rule-based judge as second opinion              #
    # ================================================================ #
    if llm.get("rule_based_judge"):
        console.print()
        console.print(Panel(
            "[bold white]The section below is entirely deterministic — no AI is involved.\n"
            "It scores the same underlying data through four explicit rule-based lenses\n"
            "and serves as an independent cross-check on the AI pipeline above.[/bold white]",
            title="[bold white]◈  DETERMINISTIC RULE-BASED JUDGE  (Second Opinion)  ◈[/bold white]",
            border_style="white",
            expand=True,
        ))
        console.print()
        display_rule_based_judge(
            llm,
            statistical=statistical,
            info=info,
            title="RULE-BASED JUDGE  ─  Deterministic Second Opinion",
        )


# ------------------------------------------------------------------ #
# Master display function
def display_ml_predictions(ml_result: dict) -> None:
    """Display ML model predictions in a rich panel."""
    status = ml_result.get("status", "")
    if status == "insufficient_data":
        console.print(Panel(
            "[yellow]Not enough historical data to train ML models "
            f"(minimum {100} labelled rows required).[/yellow]",
            title="[bold magenta]ML Predictions[/bold magenta]",
            border_style="magenta",
        ))
        return
    if status == "error" or not ml_result.get("clf_5d_direction"):
        console.print(Panel(
            "[dim]ML predictions unavailable.[/dim]",
            title="[bold magenta]ML Predictions[/bold magenta]",
            border_style="magenta",
        ))
        return

    # ── Sentiment row ─────────────────────────────────────────────── #
    sentiment   = ml_result.get("sentiment", {})
    sent_score  = sentiment.get("overall_score", 0.0)
    sent_label  = sentiment.get("label", "NEUTRAL")
    sent_summary = sentiment.get("summary", "")
    sent_colour = (
        "green"  if sent_label == "POSITIVE" else
        "red"    if sent_label == "NEGATIVE" else
        "yellow"
    )
    sent_score_str = f"{sent_score:+.2f}"

    # ── Build prediction table ──────────────────────────────────────  #
    tbl = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=False,
    )
    tbl.add_column("Horizon",           style="white",  width=10)
    tbl.add_column("Direction",         style="white",  width=10)
    tbl.add_column("P(Up)",             style="white",  width=8)
    tbl.add_column("Expected Return",   style="white",  width=16)

    for h, label in ((5, "5-Day"), (21, "21-Day")):
        direction  = ml_result.get(f"clf_{h}d_direction", "N/A")
        prob_up    = ml_result.get(f"clf_{h}d_prob_up")
        ret_pct    = ml_result.get(f"reg_{h}d_return_pct")

        dir_colour = "green" if direction == "UP" else ("red" if direction == "DOWN" else "yellow")
        dir_str    = f"[{dir_colour}]{direction}[/{dir_colour}]"
        prob_str   = f"{prob_up:.1%}" if prob_up is not None else "N/A"
        ret_str    = (
            f"[{'green' if ret_pct >= 0 else 'red'}]{ret_pct:+.2f}%[/{'green' if ret_pct >= 0 else 'red'}]"
            if ret_pct is not None else "N/A"
        )
        tbl.add_row(label, dir_str, prob_str, ret_str)

    # ── Metrics footnote ───────────────────────────────────────────── #
    metrics = ml_result.get("metrics", {})
    trained_rows = ml_result.get("trained_on_rows", 0)
    footnote_parts = []
    if trained_rows:
        footnote_parts.append(f"trained on {trained_rows} rows")
    acc_5  = metrics.get("clf_5d_accuracy")
    acc_21 = metrics.get("clf_21d_accuracy")
    mae_5  = metrics.get("reg_5d_mae")
    mae_21 = metrics.get("reg_21d_mae")
    if acc_5  is not None: footnote_parts.append(f"5d-clf acc {acc_5:.1%}")
    if acc_21 is not None: footnote_parts.append(f"21d-clf acc {acc_21:.1%}")
    if mae_5  is not None: footnote_parts.append(f"5d-reg MAE {mae_5:.4f}")
    if mae_21 is not None: footnote_parts.append(f"21d-reg MAE {mae_21:.4f}")
    footnote = "  [dim]" + " | ".join(footnote_parts) + "[/dim]" if footnote_parts else ""

    sent_line = (
        f"  Headline sentiment: [{sent_colour}]{sent_label}[/{sent_colour}] "
        f"(score {sent_score_str})"
    )
    if sent_summary:
        sent_line += f"  [dim]— {sent_summary}[/dim]"

    from rich.text import Text
    body = Text.assemble(sent_line, "\n")

    console.print(Panel(
        tbl,
        title="[bold magenta]ML Predictions (GradientBoosting)[/bold magenta]",
        border_style="magenta",
        subtitle=sent_line + footnote,
    ))


# ------------------------------------------------------------------ #

def display_full_report(
    ticker: str,
    info: dict,
    technical: dict,
    fundamental: dict,
    statistical: dict,
    analyst_data: dict,
    llm: dict,
    ml_result: dict | None = None,
    report_path: str = None,
) -> None:
    """
    Print the complete FinBot analysis report to the terminal.

    Parameters
    ----------
    ml_result   : dict | None
        Predictions dict from ml.predictor.predict().  When provided,
        renders an ML panel before the report-path footer.
    report_path : str | None
        If provided, prints a footer indicating where the report was saved.
    """
    display_header(ticker, info)
    display_basic_description(llm, info)
    display_technical(technical)
    display_fundamental(fundamental)
    display_statistical(statistical)
    display_analyst(analyst_data)
    display_llm(llm, statistical=statistical, info=info)

    if ml_result:
        display_ml_predictions(ml_result)

    if report_path:
        console.print(Rule())
        console.print(f"  [dim]Report saved to:[/dim] [bold cyan]{report_path}[/bold cyan]")
        console.print()
