# FinBot — AI-Powered Quantitative Stock Analyser

FinBot is a command-line stock analysis tool that combines classic quantitative finance (technical indicators, fundamental valuation, statistical models, Monte Carlo simulation) with a **hierarchical multi-team LLM pipeline** to produce a full investment report for any Yahoo Finance ticker. When no LLM credentials are present, a fully deterministic four-lens rule-based judge runs instead and produces the same output shape.

---

## Table of Contents

1. [What FinBot Does](#what-finbot-does)
2. [Why It Works](#why-it-works)
3. [System Overview (How It Works)](#system-overview-how-it-works)
4. [Quick Start](#quick-start)
5. [Requirements & Installation](#requirements--installation)
6. [Configuration (.env)](#configuration-env)
7. [Editing tickers.txt](#editing-tickerstxt)
8. [How Data Is Collected](#how-data-is-collected)
9. [Analysis Pipeline (Steps 1–6)](#analysis-pipeline-steps-16)
10. [The 5-Team, 13-Agent LLM System](#the-5-team-13-agent-llm-system)
    - [Architecture Overview](#architecture-overview)
    - [Team 1 — Analyst Team (4 parallel)](#team-1--analyst-team-4-parallel-agents)
    - [Team 2 — Researcher Team (3 sequential)](#team-2--researcher-team-3-sequential)
    - [Team 3 — Trading Team (3 parallel)](#team-3--trading-team-3-parallel-agents)
    - [Team 4 — Risk Management (2 parallel)](#team-4--risk-management-2-parallel-agents)
    - [The Judge (1 agent)](#the-judge-1-agent)
    - [Execution Flow](#execution-flow)
    - [ReAct Prompting](#react-prompting)
    - [Price Target Accuracy](#price-target-accuracy)
11. [Rule-Based Judge (--no-llm Mode)](#rule-based-judge----no-llm-mode)
12. [Reading the Terminal Display](#reading-the-terminal-display)
13. [Reading the Report (Markdown & PDF)](#reading-the-report-markdown--pdf)
14. [Output Files](#output-files)
15. [ML Model Accuracy](#ml-model-accuracy)
16. [Understanding Verdict Divergence](#understanding-verdict-divergence)
17. [Feedback Loop & Historical Accuracy](#feedback-loop--historical-accuracy)
18. [Disclaimer](#disclaimer)

---

## What FinBot Does

Give FinBot a ticker and it produces a complete, self-contained investment briefing:

- **Pulls everything from Yahoo Finance** (free, no data key) — 5 years of prices, ~100
  fundamental fields, analyst targets, news, earnings dates, options, and the VIX.
- **Runs the classic quant stack** — technical indicators, fundamental valuation, statistical
  risk models, and a 2,500-path Monte Carlo simulation.
- **Trains per-ticker ML models** — a GradientBoosting + RandomForest ensemble that predicts
  5-day and 21-day direction and return, validated leakage-free and calibrated against real
  outcomes.
- **Reaches a verdict two ways** — a hierarchical **5-team, 13-agent LLM pipeline** when an API
  key is present, or a fully deterministic **4-lens rule-based judge** when it isn't (same output
  shape either way).
- **Scores itself over time** — every prediction is logged, later resolved against the actual
  price, and fed back in to correct bias, calibrate ML probabilities, and ground the AI Judge's
  confidence in its real track record.
- **Delivers it three ways** — a colour-coded terminal report, a Markdown report with charts,
  and a polished multi-page PDF.

---

## Why It Works

The design leans on five ideas rather than any single magic signal:

1. **Triangulation.** Independent quant models, ML, and an LLM (or rule-based) judge look at the
   same data from different angles. Agreement across methods is a stronger signal than any one of
   them; disagreement is surfaced (see [Understanding Verdict Divergence](#understanding-verdict-divergence))
   rather than hidden.
2. **Leakage-free validation.** ML models use a strict **chronological 80/20 split** — never
   shuffled — so reported accuracy reflects predicting the *future* from the *past*, not curve-fit.
3. **Graceful degradation.** The rule-based 4-lens judge mirrors the LLM output shape, so FinBot
   produces a full, reproducible report with **zero external calls** when no key is configured.
4. **A closed feedback loop.** Predictions are measured against reality and the results flow back
   in: median bias correction on price targets, empirical probability calibration for the ML
   models, and **realized-accuracy grounding** of the AI Judge's confidence. The tool gets more
   honest the more you run it.
5. **Adversarial debate.** The LLM pipeline pits a Bull researcher against a Bear researcher and
   forces a Synthesizer to reconcile them — reducing the one-sided optimism a single prompt tends
   to produce.

> None of this predicts the market reliably — markets are largely unpredictable. The goal is a
> rigorous, transparent, *self-correcting* briefing, not a crystal ball. See the
> [Disclaimer](#disclaimer).

---

## System Overview (How It Works)

End-to-end flow for a single run. The dashed arrow is the **feedback loop** — each run's
predictions are resolved later and feed corrections back into future runs.

```
        ┌──────────────────────────────────────────────────────────────┐
        │  DATA  (Yahoo Finance via yfinance — no API key)             │
        │  5y OHLCV · fundamentals · analyst data · news · VIX · options│
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  QUANT ANALYSIS                                                    │
   │  technical.py · fundamental.py · statistical.py (+ Monte Carlo)    │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  FEATURE MATRIX  → features.csv  (33 engineered features)          │
   │            ▼                                                       │
   │  ML ENSEMBLE  GradientBoosting + RandomForest (5d & 21d)           │
   │  direction · return · calibrated probability · confidence         │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  JUDGEMENT  (pick one)                                             │
   │   • LLM: 5-team / 13-agent pipeline → The Judge   (key present)    │
   │   • Rule-based: deterministic 4-lens judge        (--no-llm)       │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  OUTPUT   terminal report · reports/{TICKER}_{DATE}/report.md +.pdf│
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  FEEDBACK LOOP                                                     │
   │  tracker → saves prediction │ resolver → fetches actual price      │
   │  accuracy → bias correction + ML calibration + Judge grounding     │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ╎ (corrections applied on the next run)
                                   └╌╌╌╌╌╌╌╌╌► back into JUDGEMENT
```

The drill-down for the LLM box is the [13-agent architecture diagram](#architecture-overview)
further down.

---

## Quick Start

> **First time?** Follow the full [Requirements & Installation](#requirements--installation) section below before running any commands.

### Step 1 — Activate the virtual environment

Every time you open a new terminal, activate the bundled venv first:

```powershell
# Windows (PowerShell)
.\finbot_venv\Scripts\Activate.ps1

# Windows (Command Prompt)
finbot_venv\Scripts\activate.bat

# macOS / Linux
source finbot_venv/bin/activate
```

You should see `(finbot_venv)` at the start of your prompt. All `python` commands below assume this is active.

### Step 2 — Run an analysis

```bash
# Analyse a single ticker (with LLM if configured, rule-based judge otherwise)
python main.py --ticker AAPL

# Skip the LLM step entirely — runs the rule-based judge (no API key required)
python main.py --ticker MSFT --no-llm

# Force ML model retraining for a ticker (models normally auto-retrain every 7 days)
python main.py --ticker TSLA --retrain

# Run every ticker listed in tickers.txt in one go
python main.py --file tickers.txt

# Same batch run, but skip LLM (faster, no API costs)
python main.py --file tickers.txt --no-llm
```

### Step 3 — Resolve past predictions (optional, run periodically)

FinBot saves every prediction it makes. Once enough time has passed (e.g. 7 days for the 1-week horizon), `--resolve` fetches the actual closing price and scores whether the prediction was correct. Run this regularly to build up accuracy history:

```bash
# Resolve all tickers that have pending predictions (scans all feedback CSVs)
python main.py --resolve

# Resolve only one ticker
python main.py --ticker AAPL --resolve

# Resolve all tickers in your watchlist file
python main.py --file tickers.txt --resolve
```

`--resolve` **does not** run a new analysis — it only fills in outcomes for old predictions. It is safe to run at any time; already-resolved horizons are skipped automatically.

### Step 4 — Bootstrap accuracy statistics (automatic on first run)

The feedback loop requires resolved predictions before bias correction and ML calibration activate. **This now happens automatically** — on the first run for any ticker, FinBot detects that the feedback CSV is empty and immediately replays the trained model over the test-split portion of `features.csv` to populate ~250 resolved rows.

You can also trigger it manually (e.g. after deleting a feedback CSV, or to force a refresh):

```bash
# Manually backfill a single ticker
python main.py --ticker SPY --backfill

# Manually backfill all tickers in your watchlist
python main.py --file tickers.txt --backfill
```

`--backfill` replays the **last 20%** of `features.csv` using temporary models fitted only on the first 80%, so the replayed window is genuinely out-of-sample and there is no look-ahead bias. Backfill rows are tagged `provider="backfill"` in the CSV.

### Step 5 — Discover new tickers (`--discover`)

Scans the market for fresh candidates using four free, key-less sources:

1. **yfinance predefined screeners** — day gainers, most actives, small-cap gainers, undervalued growth, growth tech.
2. **Volume-spike scan** — every NASDAQ/NYSE common stock whose volume is >3× its 20-day average with positive 5-day momentum.
3. **52-week-high breakouts** — closes within 2% of a fresh 52-week high on above-average volume (computed from the same download — no extra requests).
4. **New-listing detector** — diffs today's NASDAQ symbol directory against the previous cached copy, surfacing brand-new IPOs/uplistings. (Needs one prior run as a baseline.)

Candidates flagged by multiple sources rank up. Survivors of price/liquidity/market-cap filters get a quick two-lens rule-based score, and the best are appended to `watchlist.txt` with provenance comments:

```bash
# Full discovery scan (the volume scan takes a few minutes)
python main.py --discover

# Tune how many picks get added and the minimum market cap
python main.py --discover --discover-top 10 --discover-min-cap 500000000

# Promote discoveries into a full analysis (feeds the feedback loop)
python main.py --file watchlist.txt --no-llm
```

`tickers.txt` is never modified automatically — you own it. The full ranked list with per-source notes is saved to `reports/discovery/discovery_{DATE}.csv`.

### Step 6 — Train the pooled cross-ticker model (`--retrain-global`)

Per-ticker models train on only ~1,250 rows each and overfit easily; brand-new tickers can't train at all. The global model stacks **every** ticker's feature history into one pooled dataset (all features are scale-independent ratios/z-scores, so rows are comparable across tickers):

```bash
python main.py --retrain-global
```

Once trained (saved under `models/_GLOBAL/`), the predictor automatically:
- **blends** global and per-ticker probabilities, weighted by each model's validated walk-forward CV accuracy, and
- serves **global-only** predictions for tickers with too little history for their own model — which is exactly what freshly discovered tickers need.

Re-run it occasionally (e.g. after adding tickers or monthly) to refresh the pool.

### Output

Results are printed to the terminal and saved under `reports/{TICKER}_{DATE}/`:
- `report.md` — full Markdown report
- `report.pdf` — PDF version
- `features.csv` — feature matrix used by ML models

---

## Requirements & Installation

### Prerequisites

- **Python 3.11 or newer** — check with `python --version`
- A terminal open in the project root directory (`Stock_Predictor/`)
- (Optional) At least one LLM API key — see [Configuration (.env)](#configuration-env) below. Without one, the rule-based judge runs automatically.

### First-time setup

**1. Create and activate a virtual environment**

```powershell
# Windows
python -m venv finbot_venv
.\finbot_venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv finbot_venv
source finbot_venv/bin/activate
```

> The `finbot_venv/` folder is **not** committed to the repo (it is git-ignored) — create it
> yourself with the command above. **Tip:** avoid putting the project inside a cloud-synced
> folder (OneDrive/Dropbox/iCloud); background sync can corrupt a virtualenv's package metadata.
> If imports start failing oddly, delete `finbot_venv/` and recreate it, then re-run step 2.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

This installs all packages listed below. It only needs to be done once (or after pulling updates that change `requirements.txt`).

**3. Configure your `.env` file** — see the [Configuration (.env)](#configuration-env) section.

**4. Edit your watchlist** — see the [Editing tickers.txt](#editing-tickerstxt) section.

### Core dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Free Yahoo Finance data (prices, fundamentals, analyst data, news) |
| `pandas` / `numpy` / `scipy` | Numerical computation |
| `scikit-learn` | GradientBoosting + RandomForest ensemble ML models |
| `matplotlib` | Chart generation |
| `rich` | Colour terminal output |
| `openai` | Azure OpenAI and DeepSeek API client (OpenAI-compatible) |
| `google-genai` | Google Gemini API client |
| `reportlab` | PDF generation |
| `json-repair` | Robust parsing of LLM JSON output |
| `python-dotenv` | `.env` file loading |

---

## Configuration (.env)

FinBot reads secrets and settings from a `.env` file in the project root. This file is **never committed to git**.

**1. Copy the example file:**

```powershell
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

**2. Open `.env` and fill in your values.** You only need to configure the providers you intend to use — any provider whose key is left blank is silently skipped.

```ini
# ── LLM Providers (configure at least one; leave others blank to skip) ──
AZURE_OPENAI_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o          # your deployment name, not the model name
AZURE_OPENAI_API_VERSION=2024-12-01-preview

GOOGLE_API_KEY=your_google_api_key      # free key at aistudio.google.com
GOOGLE_MODEL=gemini-2.0-flash

DEEPSEEK_API_KEY=your_deepseek_api_key  # platform.deepseek.com
DEEPSEEK_MODEL=deepseek-chat            # optional, this is the default

# ── Financial constants (safe to leave as-is) ────────────────────────
RISK_FREE_RATE=0.0525
EQUITY_RISK_PREMIUM=0.055
MAX_RISK_PER_TRADE=0.01

# ── LLM tuning (safe to leave as-is) ────────────────────────────────
LLM_TEMPERATURE=0.0          # default 0.0 (deterministic); raise toward 0.3 for more variety
LLM_MAX_TOKENS_AZURE=8000
LLM_MAX_TOKENS_GOOGLE=8192
LLM_MAX_TOKENS_DEEPSEEK=8000
```

If multiple providers are configured, **all run concurrently** (in parallel threads) and their outputs are combined — total wall-clock is roughly the slowest single provider, not the sum. Each active provider appears in the terminal display and Markdown report. Any provider whose API key is absent is silently skipped — no errors are raised.

> **No API key?** Just run with `--no-llm`. The rule-based judge produces a full report with no external calls.

---

## Editing tickers.txt

`tickers.txt` is the watchlist used by `--file`. Open it in any text editor:

- **One ticker per line**, using Yahoo Finance notation.
- Lines starting with `#` are comments and are ignored.
- Blank lines are ignored.
- International tickers use Yahoo Finance suffixes (e.g. `9988.HK`, `005930.KS`, `NESN.SW`).

**Example — uncomment the tickers you want:**

```
# Large-cap US tech
AAPL
MSFT
# GOOGL   ← still commented out, will be skipped

# My positions
QUBT
NOW
```

If a ticker fails (invalid symbol, no data), FinBot logs the error and continues to the next one — the whole batch does not stop.

---

## How Data Is Collected

All data is fetched free via **Yahoo Finance** using `yfinance`. No paid data API keys are required.

### Step 1 — Price History & Fundamentals (`data/stock_fetcher.py`)

Downloads **5 years** of daily OHLCV data (~1,250 rows). Also fetches ~100 fundamental fields from `yf.Ticker.info`: market cap, sector, P/E, EPS, book value, FCF, revenue growth, ROE, beta, 52-week high/low, dividend yield, etc. Additionally fetches daily **VIX** (`^VIX`) history to populate the `vix_level` and `vix_5d_change` ML features.

### Step 2 — Analyst Data (`data/analyst_fetcher.py`)

Fetches analyst consensus, price targets (mean/median/low/high), upgrade/downgrade history, up to 10 recent news headlines, upcoming earnings date, and the live CBOE VIX value (`^VIX`).

**Company Description:** Built entirely from yfinance fields — `company_name`, `sector`, `industry`, `country`, `website`, `employees`, `founding_year`, `business_overview`, `recent_news`. No LLM call is made for this section.

---

## Analysis Pipeline (Steps 1–6)

### Step 3 — Technical Indicators (`analysis/technical.py`)

| Indicator | Formula | Purpose |
|---|---|---|
| SMA 20/50/200 | Rolling arithmetic mean | Trend direction; Golden/Death Cross |
| EMA 12/26 | Exponential weighted mean (α = 2/(n+1)) | Responsive trend signal |
| RSI-14 | 100 − 100/(1 + AvgGain/AvgLoss) | Overbought (>70) / oversold (<30) |
| MACD | EMA12 − EMA26; Signal = EMA9 | Momentum crossover |
| Bollinger Bands | SMA20 ± 2σ; %B; Bandwidth | Volatility squeeze/expansion |
| Fibonacci Retracements | High − (High−Low) × ratio | Key support/resistance levels |
| Support & Resistance Pivots | Rolling swing highs/lows (window=10) | Structural price levels |
| ATR-14 | Wilder-smoothed True Range | Daily range proxy; stop-loss sizing |
| ADX-14 | Wilder-smoothed DX from ±DM | Trend strength (>25 = strong trend) |
| OBV | Cumulative ±Volume by close direction | Volume/price divergence |

### Step 4 — Fundamental Metrics (`analysis/fundamental.py`)

| Metric | Formula |
|---|---|
| Trailing / Forward P/E | Price ÷ Trailing or Forward EPS |
| PEG Ratio | P/E ÷ Earnings Growth Rate |
| Graham Number | √(22.5 × EPS × Book Value) |
| DCF Value | FCF × (1+g)^n ÷ (WACC−g) — WACC=10%, g=3% |
| Price-to-Book | Price ÷ Book Value Per Share |
| Dividend Yield | Annual Dividend ÷ Price |

### Step 5 — Statistical Models & Monte Carlo (`analysis/statistical.py`)

| Model | Output |
|---|---|
| Historical Volatility | σ_annual = σ_daily × √252; regime label |
| Beta | Cov(R_stock, R_SPY) / Var(R_SPY) |
| Sharpe & Sortino Ratios | Risk-adjusted return quality |
| Max Drawdown & Calmar | Tail-risk and recovery quality |
| OLS Trend Regression | Extrapolated price at 8 horizons with R² |
| Monte Carlo GBM | 2,500 paths; P10/Median/P90 at 8 horizons |

Monte Carlo drift (μ) blends 60% CAPM-implied return (Rf + β × ERP) and 40% historical mean log-return. Volatility (σ) blends 70% long-term (5-year) and 30% recent (30-day). **2,500 simulation paths** are used (up from 1,000) for tighter confidence intervals.

### Step 6 — ML Predictions (`ml/trainer.py`, `ml/predictor.py`)

Two model families trained per ticker on **33 features** (returns, MA ratios, oscillators, volume, sentiment, VIX macro context, earnings proximity, and price z-score). Each horizon (5-day and 21-day) has a **GradientBoosting + RandomForest ensemble** — their `predict_proba` outputs are averaged for more stable predictions. Produces directional probability, calibrated probability (once feedback data accumulates), confidence label (HIGH/MEDIUM/LOW), and expected return %. Chronological 80/20 split; models cached under `models/{TICKER}/` and auto-retrained when either (a) models are older than 7 days, or (b) `FEATURE_COLS` has changed since the last training run (e.g. after a code upgrade adding new features).

> **Note on `sentiment_score`:** this feature only carries real data on the **last row** (today's headlines, LLM-scored at run time). All historical rows default to `0.0` (NEUTRAL). The model therefore learns when sentiment is neutral, and detects deviations on the current prediction row.

**New features added in v1.2.0:**

| Feature | Source | Signal type |
|---|---|---|
| `vix_level` | `^VIX` daily close | Market fear regime |
| `vix_5d_change` | VIX 5-day % change | Fear direction |
| `earnings_within_14d` | `yf.Ticker.get_earnings_dates()` | Event risk flag |
| `price_zscore_20d` | `(Close − SMA_20) / σ_20` | Mean-reversion signal |

---

## The 5-Team, 13-Agent LLM System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER  (yfinance — no LLM)                            │
│  price_df · info · technical · fundamental · statistical    │
│  analyst_data · ml_result · company_description             │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  STEP 1 — TEAM 1: ANALYST TEAM  │  4 agents in parallel
          │  A1 Fundamental Analyst          │
          │  A2 Sentiment Analyst            │
          │  A3 News Analyst                 │
          │  A4 Technical Analyst            │
          └────────────────┬────────────────┘
                           │  analyst reports
          ┌────────────────▼────────────────┐
          │  STEP 2 — TEAM 2: RESEARCHERS   │  Bull + Bear in parallel
          │  R1 Bullish Researcher           │
          │  R2 Bearish Researcher           │
          └────────────────┬────────────────┘
                           │  bull + bear cases
          ┌────────────────▼────────────────┐
          │  STEP 3 — SYNTHESIZER           │  1 agent (serial)
          │  R3 Research Synthesizer         │
          └────────────────┬────────────────┘
                           │  balanced synthesis
          ┌────────────────▼────────────────┐
          │  STEP 4 — TEAM 3 + RM1          │  4 agents in parallel
          │  T1 Momentum Trader              │
          │  T2 Value Trader                 │
          │  T3 Swing Trader                 │
          │  RM1 Market Risk Agent           │
          └────────────────┬────────────────┘
                           │  trade plans + market risk
          ┌────────────────▼────────────────┐
          │  STEP 5 — RM2 PORTFOLIO RISK    │  1 agent (serial)
          │  RM2 Portfolio Risk Agent        │
          └────────────────┬────────────────┘
                           │  validated sizing
          ┌────────────────▼────────────────┐
          │  STEP 6 — THE JUDGE             │  1 agent (serial)
          │  Final recommendation + targets  │
          └─────────────────────────────────┘
```

**Total agents:** 13  
**Parallel execution steps:** Steps 1, 2, and 4 use `ThreadPoolExecutor` for parallel agent calls.

---

### Team 1 — Analyst Team (4 parallel agents)

All four agents receive the same raw data simultaneously and produce independent specialist reports.

#### A1 — Fundamental Analyst

Evaluates the business on four dimensions:

| Dimension | Output fields |
|---|---|
| Business health | `business_health`: STRONG / ADEQUATE / WEAK |
| Valuation stance | `valuation_stance`: UNDERVALUED / FAIRLY_VALUED / OVERVALUED |
| Capital allocation | `capital_allocation`: EXCELLENT / ADEQUATE / POOR |
| Profitability quality | `profitability_quality`: HIGH / MEDIUM / LOW |

Also produces:
- `short_term_verdict` / `long_term_verdict`: WORTH_INVESTING / NOT_WORTH_INVESTING / NEUTRAL
- `key_strengths` (list), `key_concerns` (list)
- `analyst_conviction`: based on upgrade/downgrade history and target spread

#### A2 — Sentiment Analyst

Analyses news headlines and analyst momentum for behavioural signals:

| Output field | Values |
|---|---|
| `sentiment_bias` | BULLISH / BEARISH / NEUTRAL |
| `market_mood` | EUPHORIC / OPTIMISTIC / NEUTRAL / PESSIMISTIC / FEARFUL |
| `behavioural_pressure` | BUYING_PRESSURE / SELLING_PRESSURE / NEUTRAL |
| `sentiment_score` | Float −1.0 to +1.0 |
| `earnings_risk_flag` | Boolean — true if earnings within 5 days |

#### A3 — News Analyst

Evaluates the macro and timing context from the VIX and news:

| Output field | Values |
|---|---|
| `vix_regime` | CALM (<15) / ELEVATED (15–25) / HIGH_FEAR (>25) / EXTREME_FEAR (>35) |
| `macro_bias` | RISK_ON / RISK_OFF / NEUTRAL |
| `timing_caution` | NONE / MINOR / MAJOR |
| `sector_timing` | BULLISH / NEUTRAL / BEARISH |
| `override_short_term` | Boolean — forces the Judge to reduce ST confidence |

#### A4 — Technical Analyst

Absorbs the full statistical and technical picture:

| Output field | Description |
|---|---|
| `statistical_bias` | BULLISH / BEARISH / NEUTRAL derived from indicator signals |
| `worth_investing` | YES / NO / CONDITIONAL |
| `direction` | LONG / SHORT |
| `entry_price`, `stop_loss` | Technically derived levels |
| `price_predictions` | Nested dict: `short_term` (1W/2W/3W/1M) and `long_term` (3M/6M/9M/12M), each with `price`, `change_pct`, `method`, `confidence`, `rationale` |

---

### Team 2 — Researcher Team (3 sequential)

#### R1 — Bullish Researcher

Receives all Analyst Team outputs. Constructs the strongest possible bull case:

- `bull_stance`: STRONG / MODERATE / WEAK
- `bull_thesis`: concise thesis string
- `growth_catalysts`: list of specific forward-looking catalysts
- `upside_scenarios`: base and optimistic price targets
- `bull_confidence`: HIGH / MEDIUM / LOW

#### R2 — Bearish Researcher

Runs in parallel with R1. Constructs the strongest possible bear case:

- `bear_stance`: STRONG / MODERATE / WEAK
- `bear_thesis`: concise thesis string
- `key_risks`: list of specific risks
- `downside_scenarios`: base and pessimistic price targets
- `bear_confidence`: HIGH / MEDIUM / LOW

#### R3 — Research Synthesizer

Receives R1 and R2 outputs. Adjudicates the debate and produces a balanced view:

- `net_bias`: BULLISH / BEARISH / NEUTRAL
- `consensus_strength`: STRONG / MODERATE / WEAK
- `agreed_points`: list of points both researchers accept
- `disputed_points`: list of genuine disagreements
- `balanced_brief`: one-paragraph synthesis

---

### Team 3 — Trading Team (3 parallel agents)

All three traders receive the full analyst + researcher context. Each specialises in a different style and time horizon.

#### T1 — Momentum Trader (short-term focus)

Targets 1–4 week horizons. Reads RSI momentum, MACD crossovers, ADX trend strength, and recent price action.

| Output | Description |
|---|---|
| `trade_action` | ENTER_LONG / ENTER_SHORT / HOLD / WAIT / EXIT |
| `entry_price`, `stop_loss` | Momentum-driven levels |
| `target_1w`, `target_2w`, `target_3w`, `target_1m` | Price targets |
| `position_size_pct` | As % of account |
| `momentum_score` | STRONG / MODERATE / WEAK |

#### T2 — Value Trader (medium-to-long-term focus)

Targets 3–12 month horizons. Weights DCF, Graham Number, margin of safety, and analyst price target upside.

| Output | Description |
|---|---|
| `trade_action` | ENTER_LONG / ENTER_SHORT / HOLD / WAIT |
| `entry_price`, `stop_loss` | Fundamental-anchored levels |
| `target_3m`, `target_6m`, `target_9m`, `target_12m` | Price targets |
| `margin_of_safety_pct` | Buffer below intrinsic value estimate |
| `value_score` | DEEP_VALUE / FAIR_VALUE / OVERVALUED |

#### T3 — Swing Trader (pattern-based)

Targets 1–4 week mean-reversion or breakout setups. Reads Bollinger %B, support/resistance pivots, and Fibonacci levels.

| Output | Description |
|---|---|
| `trade_action` | ENTER_LONG / ENTER_SHORT / HOLD / WAIT |
| `entry_price`, `stop_loss` | Swing-level entries |
| `target_1w`, `target_2w`, `target_3w`, `target_1m` | Price targets |
| `swing_setup` | e.g. BOLLINGER_SQUEEZE / FIBONACCI_BOUNCE / NONE |
| `nearest_support`, `nearest_resistance` | Key pivot levels |

---

### Team 4 — Risk Management (2 parallel agents)

#### RM1 — Market Risk Agent

Runs in parallel with the Trading Team in Step 4. Evaluates external risk factors:

| Output | Values |
|---|---|
| `market_risk_level` | LOW / MODERATE / HIGH / EXTREME |
| `vix_risk_flag` | CLEAR / CAUTION / DANGER |
| `earnings_caution` | NONE / REDUCE_SIZE / AVOID |
| `timing_recommendation` | PROCEED / REDUCE / AVOID |
| `max_position_flag` | FULL_SIZE / HALF_SIZE / NO_TRADE |
| `liquidity_risk` | LOW / MEDIUM / HIGH |

#### RM2 — Portfolio Risk Agent

Runs in Step 5 (serial) after receiving Trading Team outputs and RM1's market risk verdict. Validates and finalises position sizing:

| Output | Description |
|---|---|
| `kelly_fraction` | Full Kelly position sizing fraction |
| `recommended_size_pct` | Final recommended size as % of account |
| `sizing_method` | ATR_BASED / KELLY / FIXED |
| `scaling_factor` | FULL / NORMAL / HALF |
| `validated_entry`, `validated_stop` | Confirmed after cross-checking traders |
| `stop_validation` | VALID / ADJUSTED / REJECTED |
| `stop_validation_note` | Reason if stop was adjusted or rejected |

**Scaling rules applied by RM2:**

| Condition | Size |
|---|---|
| HIGH confidence + CALM VIX + no earnings risk | 1.5× normal |
| MEDIUM confidence or ELEVATED VIX | 1.0× normal |
| LOW confidence or HIGH_FEAR VIX or earnings ≤5 days | 0.5× normal |
| Maximum regardless of confidence | 5% of portfolio |

---

### The Judge (1 agent)

The Judge runs last (Step 6, serial). It receives every team's structured output plus the raw data and ML signals, and synthesises a single final decision.

**Context injected into the Judge prompt:**

```
=== COMPANY DESCRIPTION ===    yfinance metadata (no LLM)
=== ANALYST TEAM ===           A1/A2/A3/A4 reports
=== RESEARCHER TEAM ===        R1 bull case, R2 bear case, R3 synthesis
=== TRADING TEAM ===           T1/T2/T3 trade plans
=== RISK MANAGEMENT ===        RM1 market risk, RM2 validated sizing
=== ML MODEL SIGNALS ===       5d/21d direction probability + predicted return%
=== EARNINGS CALENDAR ===      days to earnings, risk level
=== VIX & MACRO ===            current VIX value and regime
=== ORIGINAL DATA ===          raw technicals, stats, fundamentals
```

**Judge output:**

| Field | Description |
|---|---|
| `recommendation` | STRONG BUY / BUY / HOLD / SELL / STRONG SELL |
| `confidence` | HIGH / MEDIUM / LOW |
| `overall_short_term` | WORTH_INVESTING / NOT_WORTH_INVESTING / NEUTRAL |
| `overall_long_term` | WORTH_INVESTING / NOT_WORTH_INVESTING / NEUTRAL |
| `target_prices` | 8-horizon dict — see below |
| `entry_price`, `exit_price`, `stop_loss` | Final trading levels |
| `position_size_pct` | Validated position size |
| `trade_direction` | LONG / SHORT |
| `key_bull_case`, `key_bear_case` | Lists |
| `key_risks`, `catalysts` | Lists |
| `summary` | One-paragraph overall verdict |
| `technical_verdict`, `fundamental_verdict`, `valuation_verdict`, `sentiment_verdict`, `macro_verdict` | Per-dimension one-liners |
| `alternative_pick` | Optional better ticker suggestion |

---

### Price Target Structure

Price targets are returned at 8 horizons. Each horizon is a dict, not a flat float:

```python
target_prices = {
    "1_week":   {"price": 142.50, "accuracy_pct": 61.2},
    "2_weeks":  {"price": 145.00, "accuracy_pct": 58.1},
    "3_weeks":  {"price": 147.20, "accuracy_pct": 55.3},
    "1_month":  {"price": 150.00, "accuracy_pct": 51.8},
    "3_months": {"price": 158.00, "accuracy_pct": 44.0},
    "6_months": {"price": 165.00, "accuracy_pct": 38.5},
    "9_months": {"price": 170.00, "accuracy_pct": 35.0},
    "12_months":{"price": 175.00, "accuracy_pct": 32.5},
}
```

`accuracy_pct` is a per-horizon estimated prediction accuracy. Once the feedback loop has a
track record for the ticker (N ≥ `FEEDBACK_MIN_SAMPLES` resolved predictions at a horizon), the
Judge is instructed to use that horizon's **realized directional accuracy** as the base. Until
then it falls back to the heuristic below (ML classifier test accuracy × decay × VIX penalty).
Longer horizons have lower accuracy — this is by design.

---

### Execution Flow

```
Step 1 — 4 parallel agents   →  Analyst Team (A1, A2, A3, A4)
Step 2 — 2 parallel agents   →  Bull + Bear Researchers (R1, R2)
Step 3 — 1 serial agent      →  Research Synthesizer (R3)
Step 4 — 4 parallel agents   →  Traders (T1, T2, T3) + Market Risk (RM1)
Step 5 — 1 serial agent      →  Portfolio Risk (RM2)
Step 6 — 1 serial agent      →  The Judge
```

Each parallel step uses `ThreadPoolExecutor`. The LLM provider callable (`_call_azure`,
`_call_google`, or `_call_deepseek`) is passed as a function argument so the same orchestrator
works for every provider.

**Two layers of parallelism + client reuse.** When more than one provider is configured, the
**whole 6-step chain runs concurrently for each provider** (Azure, Gemini, DeepSeek) in its own
thread — so adding a second or third provider barely changes wall-clock. Within each chain, the
steps above parallelise their agents. Each provider's API client is built **once** and cached
(`_get_azure_client` / `_get_google_client` / `_get_deepseek_client` via `lru_cache`) and reused
across all 13 agent calls instead of being reconstructed every call.

---

### ReAct Prompting

Every agent uses a **ReAct** (Reason + Act) system prompt. The agent is instructed to show its reasoning chain before producing its final answer:

```
Thought: <reasoning about the inputs>
Action: analyse the data
Observation: <what the data shows>
... (repeat as needed)
ANSWER:
{ ...JSON output... }
```

The `_extract_json()` function strips everything before `ANSWER:` before parsing. This makes the JSON extraction robust against agents that include lengthy reasoning preambles.

---

### Price Target Accuracy

`accuracy_pct` in each target uses a **realized-first** base:

```
IF the ticker has a track record at this horizon (N ≥ FEEDBACK_MIN_SAMPLES resolved):
    base_accuracy = realized directional accuracy for that horizon   ← preferred
ELSE (no track record yet — cold start):
    base_accuracy = ML classifier test accuracy for the relevant horizon
                    (clf_5d for ≤1 month, clf_21d for ≥3 months)
    decay_factor  = {1W: 1.00, 2W: 0.95, 3W: 0.90, 1M: 0.85,
                     3M: 0.75, 6M: 0.65, 9M: 0.60, 12M: 0.55}
    base_accuracy = base_accuracy × decay_factor

vix_penalty    = −5% for HIGH_FEAR, −2% for ELEVATED, 0% for CALM
accuracy_pct   = clip(base_accuracy + adjustments + vix_penalty, 30, 85)
```

**Confidence caution rules** (injected alongside the realized stats): if the ticker's overall
recommendation hit-rate is below 50% the Judge caps `confidence` at MEDIUM, below 40% at LOW;
horizons whose realized directional accuracy is worse than a coin flip are flagged low-confidence.
This is what keeps the headline numbers honest instead of persistently over-confident.

---

## Rule-Based Judge (`--no-llm` Mode)

When `--no-llm` is passed or no LLM credentials are configured, `run_rule_based_analysis()` runs instead of the agent chain. It produces a dict with the **same shape** as `get_llm_analysis()` — including all 12 agent stubs, `target_prices` with `accuracy_pct`, `trade_direction`, `entry/exit/stop`, `position_size_pct`, and all verdict fields — so the terminal display and report generator work identically in both modes.

Internally, it calls `run_rule_based_judge()`, which scores four deterministic lenses:

### Lens 1 — Fundamental: "Is the business healthy?"

7 signals scored 0–2 points each. Verdict: HEALTHY (≥66%) / ADEQUATE (45–65%) / WEAK (<45%).

| Signal | Threshold |
|---|---|
| ROE | Strong >15%: 2pt · Adequate 8–15%: 1pt · Weak <8%: 0pt |
| Operating Margin | Strong >20%: 2pt · Moderate 10–20%: 1pt · Low <10%: 0pt |
| Net Profit Margin | High >15%: 2pt · Moderate 5–15%: 1pt · Low <5%: 0pt |
| Revenue Growth | Strong >10%: 2pt · Positive 0–10%: 1pt · Declining <0%: 0pt |
| FCF Yield (FCF ÷ Mkt Cap) | High >5%: 2pt · Moderate 2–5%: 1pt · Negative: 0pt |
| Interest Coverage (OpCF ÷ Interest) | Strong >3.0×: 2pt · Adequate 1.5–3.0×: 1pt · Weak: 0pt |
| Debt/Equity | Low <1.0×: 2pt · Moderate 1.0–2.5×: 1pt · High >2.5×: 0pt |

### Lens 2 — Technical: "Is the price/timing right?"

8 signals. Verdict: BULLISH (≥66%) / NEUTRAL (45–65%) / BEARISH (<45%).

| Signal | Scoring |
|---|---|
| RSI-14 | Entry zone 30–55: 2pt · Oversold/Elevated: 1pt · Overbought >70: 0pt |
| MACD | Bullish crossover: 2pt · Neutral: 1pt · Bearish: 0pt |
| Price vs SMA 20/50/200 | Above all: 2pt · Majority: 1pt · Below majority: 0pt |
| ADX + ±DI | Strong bullish (ADX>25 & PDI>NDI): 2pt · Ranging: 1pt · Strong bearish: 0pt |
| Bollinger %B | Lower/mid band: 2pt · Upper half: 1pt · Extended >0.85: 0pt |
| 52-Week Position | Near high or near low: 1pt · Mid-range: 0pt |
| OBV | Bullish/rising: 2pt · Neutral: 1pt · Bearish/diverging: 0pt |
| OLS Trend | Upward: 1pt (2pt if R²>0.70) · Downtrend: 0pt |

### Lens 3 — Valuation: "Am I overpaying?"

Up to 9 signals. Verdict: UNDERVALUED (≥66%) / FAIRLY VALUED (45–65%) / OVERVALUED (<45%).

**Sector-relative grading:** trailing P/E, forward P/E and P/B are compared to the stock's
**own sector-peer medians** (top ~100 US names in the sector by market cap, one key-free
Yahoo screener request per sector, cached 7 days in `.cache/sector_medians.json`). A 30× P/E
is cheap for semiconductors and expensive for banks — absolute bands punished growth sectors
and flattered deep value. When sector data is unavailable (ETFs, network failure with no
cache), the absolute thresholds below apply.

| Signal | Threshold |
|---|---|
| Trailing P/E | vs sector median — <0.8×: 2pt · 0.8–1.2×: 1pt · >1.2×: 0pt · (fallback: <15×: 2pt · 15–25×: 1pt · >40×: 0pt) |
| Forward P/E | vs sector median — same bands (fallback: <15×: 2pt · 15–25×: 1pt · >25×: 0pt) |
| PEG Ratio | <1.0: 2pt · 1.0–2.0: 1pt · >2.0: 0pt |
| EV/EBITDA | <10×: 2pt · 10–20×: 1pt · >20×: 0pt |
| DCF vs Price | Undervalued: 2pt · Within 20%: 1pt · Overvalued: 0pt |
| Graham Number vs Price | Below: 2pt · Slightly above: 1pt · Well above: 0pt |
| Price/Book | vs sector median — same bands (fallback: <1.5×: 2pt · 1.5–3.0×: 1pt · >3.0×: 0pt) |
| Analyst Mean Target | >5% upside: 1pt · At/below target: 0pt |

### Lens 4 — Risk: "What is the downside exposure?"

5 signals. Verdict: LOW RISK (≥66%) / MODERATE RISK (45–65%) / HIGH RISK (<45%).

| Signal | Threshold |
|---|---|
| Sortino Ratio | >2.0: 2pt · >1.0: 1pt · <1.0: 0pt |
| Calmar Ratio | >1.0: 2pt · 0.5–1.0: 1pt · <0.5: 0pt |
| Beta | 0.5–1.5: 2pt · Outside range: 1pt · Inverse or >2.0: 0pt |
| Volatility Regime | Compressing: 2pt · Stable: 1pt · Expanding: 0pt |
| Earnings Proximity | None near-term: 2pt · Within 21 days: 1pt · Within 5 days: 0pt |

### Composite Scoring

$$\text{composite} = 0.30 \times F + 0.25 \times T + 0.30 \times V + 0.15 \times R$$

**ML modifier:** ±10 points based on average directional probability vs 0.50 baseline.

**Hard vetoes** (cap composite at 44 regardless of score):
1. Declining revenue + negative FCF + fundamental score <35
2. BEARISH technicals + WEAK fundamentals + OVERVALUED valuation

| Composite Score | Recommendation |
|---|---|
| ≥75 | STRONG BUY |
| 60–74 | BUY |
| 45–59 | HOLD |
| 30–44 | SELL |
| <30 | STRONG SELL |

Missing data signals are skipped and the lens score is normalised over available points only — ETFs and REITs are not unfairly penalised.

---

## Reading the Terminal Display

The terminal report is **bottom-line-up-front** and **colour-coded by domain** so you can scan it
top to bottom. Read it in this order:

1. **Header panel** — company name, ticker, current price, 52-week high/low, sector/industry,
   market cap, and run date.

2. **◆ Executive Summary ◆ (read this first).** A single panel with the whole verdict:
   - **Recommendation** + **Confidence** — the panel border is colour-coded
     (**green** = BUY family · **red** = SELL family · **yellow** = HOLD/neutral).
   - **Price → key target** with the expected % move (1-month target, or 12-month if no 1-month).
   - **Outlook** — short-term and long-term verdicts (LLM mode).
   - **ML (21d)** — a one-line model cross-check (direction + probability).
   - **Top Bull** point and **Top Risk**, plus a one-paragraph summary.
   - The subtitle shows the source: *AI Analyst Pipeline* or *Rule-Based Judge (no LLM)*.

3. **Colour-coded analysis sections.** Each domain divider has its **own accent colour** so you
   always know which section you're in:

   | Section | Accent |
   |---|---|
   | Company Description | bright cyan |
   | Technical Analysis | blue |
   | Fundamental Analysis | green |
   | Statistical Analysis | cyan |
   | Analyst Consensus | yellow |
   | ML Predictions | magenta |
   | Historical Accuracy | bright yellow |

   Inside tables, **signal cells follow a fixed convention**: 🟩 green = bullish/good,
   🟥 red = bearish/risk, 🟨 yellow = neutral/caution, dim = N/A.

4. **AI Analyst Pipeline** (LLM mode) — a cyan banner marks where AI-generated content begins,
   then one block per provider showing the 4 teams (Analyst → Researcher → Trading → Risk) as
   per-agent panels, ending with **The Judge** summary. In `--no-llm` mode you instead get the
   white-bannered **deterministic rule-based judge** (four lens panels side-by-side + a Price
   Outlook table). When an LLM ran, the rule-based judge is also shown as a *second opinion*.

5. **ML Predictions panel** — 5-day and 21-day direction, P(up), and expected return, with a
   model-quality footnote (training rows, classifier accuracy, regressor MAE).

6. **Historical Accuracy panel** — appears once the feedback loop has resolved data: per-horizon
   directional accuracy, mean bias, and any active correction factors.

7. **Footer** — the path to the saved Markdown report.

> If symbols like σ/β ever look garbled, you're on a legacy code page — FinBot forces UTF-8 on
> startup, but piping into another tool may still re-encode. Viewing directly in the terminal is
> always correct.

---

## Reading the Report (Markdown & PDF)

Every run also writes a shareable report to `reports/{TICKER}_{YYYY-MM-DD}/`.

### `report.md` — Markdown (best for reading/diffing)

It opens with the bottom-line, then drills down:

1. **Executive Summary** table — recommendation, confidence, current price, key target with
   % move, outlook, top bull point, top risk, source — plus a one-line summary quote.
2. **Contents** — a clickable table of contents that jumps to each section.
3. **Company Overview** → **Price History & Moving Averages** (embedded `price_chart.png`).
4. **Technical** → **Fundamental** → **Statistical** (embedded `monte_carlo_chart.png` +
   `technical_chart.png`) — each a metric/signal table.
5. **Analyst Consensus** — consensus, targets, upgrade/downgrade history, and a collapsible
   *Legend* explaining every action and grade term.
6. **ML Predictions** and **Historical Accuracy (Feedback Loop)** tables.
7. **AI Analyst Recommendation** — one subsection per provider with all 13 agents, the Judge's
   verdict, trading levels, 8-horizon price targets, bull/bear/risks/catalysts, and a collapsible
   **Raw Judge JSON** block for the full machine-readable output.
8. **Disclaimer** footer.

Three PNG charts are saved next to the report and embedded inline: `price_chart.png`,
`technical_chart.png`, `monte_carlo_chart.png`.

### `report.pdf` — PDF (best for sharing/printing)

A polished multi-page document: a **cover page** with the recommendation badge, an **executive
summary**, then Technical / Fundamental / Statistical sections where each indicator carries its
**formula, academic source, rationale, and current value + signal**, the embedded charts, a
**mathematical appendix** with every formula, and the disclaimer. Same analysis as the Markdown
report, formatted for a reader who wants the full reasoning and citations.

---

## Output Files

Each run saves to `reports/{TICKER}_{YYYY-MM-DD}/`:

| File | Contents |
|---|---|
| `report.md` | Full Markdown report |
| `report.pdf` | PDF version |
| `features.csv` | Feature matrix used for ML training |

The feedback loop additionally writes/updates:

| File | Contents |
|---|---|
| `feedback/{TICKER}_feedback.csv` | Per-ticker prediction log — one row per provider per run, outcomes filled in over time as horizons pass |

---

## ML Model Accuracy

Models are trained per ticker on **~5 years** of daily data with an 80/20 chronological split. Each horizon uses a **GradientBoosting + RandomForest ensemble** (averaged probabilities). After each training run, the top-10 feature importances per model are saved to `models/{TICKER}/metrics.json`.

Once the feedback loop has accumulated N ≥ 10 resolved predictions, a **calibration table** is built at `models/{TICKER}/calibration.json`. This maps raw model probabilities to empirical accuracy per probability bucket, giving more reliable confidence estimates. The calibrated probability is shown alongside the raw probability in reports.

**Confidence labels** are derived from the calibrated probability:

| Label | Condition |
|---|---|
| HIGH | \|calibrated\_prob − 0.5\| > 0.20 |
| MEDIUM | \|calibrated\_prob − 0.5\| > 0.10 |
| LOW | \|calibrated\_prob − 0.5\| ≤ 0.10 |

**What accuracy to expect:** 5-day directional classification typically falls in the **50–58%** range (vs 50% random baseline). The 21-day classifier is generally stronger on large-caps and ETFs (~55–65%). Higher-volatility small-caps have wider ranges in both directions. Current per-ticker accuracy is always visible in the terminal report's ML panel and in `models/{TICKER}/metrics.json`.

**Factors that improve accuracy over time:**

| Factor | When it activates |
|---|---|
| More training data | Immediately — 5 years of history vs 2 years |
| GBM + RF ensemble | Immediately — reduces variance on erratic signals |
| 4 new features (VIX, earnings, z-score) | Immediately on retrain |
| Probability calibration | After N ≥ 10 resolved feedback rows |
| LLM bias correction | After N ≥ 5 real LLM runs resolve |
| Lens weight adjustment | After N ≥ 5 resolved rows (rule-based mode) |

---

## Understanding Verdict Divergence

When run with an LLM, both the AI pipeline and the rule-based judge produce a verdict. They will sometimes disagree — this section explains why.

### Why the rule-based judge is structurally conservative

Fixed thresholds (P/E >40× = 0 points, ROE <8% = 0 points, negative FCF = 0 points) are calibrated for profitable mature businesses. They systematically penalise early-stage or pre-profitability companies — quantum computing, biotech, SaaS at scale-up stage — where those metrics are structurally expected. The LLM pipeline will typically be more bullish on such stocks because it can read *why* the numbers look the way they do.

### Why the LLM pipeline can be structurally optimistic

LLMs are trained on financial text that skews bullish (analyst reports rarely carry strong sell ratings). The Judge inherits qualitative framing from the analyst and researcher teams — a positive narrative from the Bullish Researcher can influence the Judge even when the numbers are deteriorating.

### Reading a divergence

| Pattern | Likely meaning |
|---|---|
| LLM BUY, rule-based SELL | Thesis is entirely forward-looking. Current financials do not support the price. High-risk, high-conviction trade. |
| LLM SELL, rule-based BUY | Numbers look cheap but the LLM identified macro, competitive, or management issues the metrics miss. Investigate before buying the cheapness. |
| Both BUY | Strongest signal — fundamentals and qualitative picture both positive. |
| Both SELL | Strongest warning — numbers are bad and the qualitative picture is negative. |
| LLM BUY (LOW confidence) + rule-based SELL | The LLM is uncertain itself. Treat as hold or avoid. |

A large gap between the two verdicts is a direct measure of **how much of the bull case depends on things that cannot be verified from current financial data**.

---

## Feedback Loop & Historical Accuracy

FinBot tracks every prediction it makes and auto-resolves outcomes once each forecast horizon has elapsed.

### How It Works

1. **Prediction logging** — after each full pipeline run, a row is appended to `feedback/{TICKER}_feedback.csv` capturing the ticker, date, mode, current price, all LLM/rule-based price targets, ML signals, analyst data, volume, options metrics (put/call ratio, average IV), short interest, and all headline context.

2. **Outcome resolution** (`--resolve` flag) — fetches the actual closing price for each expired horizon (1W, 2W, 3W, 1M, 3M, 6M, 9M, 12M) and fills in `actual_price`, `pct_error`, and `dir_correct` (1 if the direction was correct, 0 otherwise). Resolution downloads each ticker's price history **once per run** and resolves every row/horizon locally (previously it made one network call per horizon per row).

3. **Bias correction** — once N ≥ `FEEDBACK_MIN_SAMPLES` (default 5) resolved rows exist for a horizon, a `correction_factor` is computed from the **median** signed error and **shrunk** on thin samples:
   ```
   correction_factor = 1 + clamp(median_bias / 100, ±FEEDBACK_MAX_BIAS_CORRECTION)
                          × min(1, N / (2 × FEEDBACK_MIN_SAMPLES))
   ```
   The median (instead of the mean) makes the correction robust to a single volatile-horizon
   outlier; the shrinkage term means 5 data points apply only half the correction, ramping to
   full strength at N ≥ 10. This multiplier is applied to LLM price targets before display and
   in the Markdown report.

4. **Dynamic lens weights** (rule-based mode) — each lens's (fundamental/technical/valuation/risk) contribution is scaled based on how accurate it has been relative to a 50% baseline, then renormalised. Capped at ±50% of the original weight. Activates at N ≥ 5 resolved rows.

5. **LLM prompt injection** — the Judge agent receives a `[HISTORICAL ACCURACY FOR {TICKER}]` block summarising per-horizon direction accuracy, mean bias, and recommendation hit rate.

### Historical Backfill (`--backfill`)

The bias correction, weight adjustment, and ML calibration all require resolved predictions to activate. Normally you would need to wait several weeks for real runs to accumulate. The backfill solves this cold-start problem.

**How it works:**
1. Loads `features.csv` for the ticker (which already contains historical feature rows **and** forward-return labels).
2. Takes only the **last 20% of labelled rows** (the test split) — rows the model was never trained on.
3. For each row on date D, runs the trained models to get predictions, then looks ahead in the price history to resolve actual prices at all 8 horizons.
4. Writes fully-resolved feedback rows tagged `mode="backfill"`, `provider="backfill"`.

With 5 years of history, the test split gives ~250 resolved rows per ticker — enough to immediately activate bias correction (N≥5), weight adjustment (N≥5), and calibration (N≥10).

**What backfill provides:**
- `ml_5d_correct` / `ml_21d_correct` → ML probability calibration
- `dir_correct_{h}` for all 8 horizons → directional accuracy stats and weight adjustment
- `actual_{h}` prices → used by resolver for overlapping real runs

**What backfill cannot provide:**
- LLM price target accuracy (`pct_error_{h}`) — no LLM ran on historical dates, so `mean_bias` and `correction_factor` still require real LLM runs.

Backfill also runs automatically (once) whenever a normal analysis detects that the feedback CSV has fewer than `FEEDBACK_MIN_SAMPLES` resolved rows.

### Horizon → Calendar Days

| Code | Calendar days |
|------|--------------|
| 1W   | 7            |
| 2W   | 14           |
| 3W   | 21           |
| 1M   | 31           |
| 3M   | 93           |
| 6M   | 186          |
| 9M   | 279          |
| 12M  | 365          |

### Configuration

| Env variable | Default | Description |
|---|---|---|
| `FEEDBACK_MIN_SAMPLES` | `5` | Minimum resolved rows before corrections/weight adjustments activate |
| `FEEDBACK_MAX_BIAS_CORRECTION` | `0.20` | Maximum bias correction clamped to ±20% |
| `BACKFILL_MIN_TEST_ROWS` | `20` | Minimum test-split rows required before backfill writes rows |
| `CALIBRATION_MIN_SAMPLES` | `10` | Minimum resolved rows before the ML probability calibration table is built |

---

## Disclaimer

FinBot is a research and educational tool. Nothing it produces constitutes financial advice. All outputs — including LLM recommendations, price targets, position sizes, and ML predictions — are generated automatically and may be wrong. Always do your own research before making any investment decision.
