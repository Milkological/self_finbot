# FinBot — AI-Powered Quantitative Stock Analyser

FinBot is a command-line stock analysis tool that combines classic quantitative finance (technical indicators, fundamental valuation, statistical models, Monte Carlo simulation) with a **hierarchical multi-team LLM pipeline** to produce a full investment report for any Yahoo Finance ticker. When no LLM credentials are present, a fully deterministic four-lens rule-based judge runs instead and produces the same output shape.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements & Installation](#requirements--installation)
3. [Configuration (.env)](#configuration-env)
4. [How Data Is Collected](#how-data-is-collected)
5. [Analysis Pipeline (Steps 1–6)](#analysis-pipeline-steps-16)
6. [The 5-Team, 13-Agent LLM System](#the-5-team-13-agent-llm-system)
   - [Architecture Overview](#architecture-overview)
   - [Team 1 — Analyst Team (4 parallel)](#team-1--analyst-team-4-parallel-agents)
   - [Team 2 — Researcher Team (3 sequential)](#team-2--researcher-team-3-sequential)
   - [Team 3 — Trading Team (3 parallel)](#team-3--trading-team-3-parallel-agents)
   - [Team 4 — Risk Management (2 parallel)](#team-4--risk-management-2-parallel-agents)
   - [The Judge (1 agent)](#the-judge-1-agent)
   - [Execution Flow](#execution-flow)
   - [ReAct Prompting](#react-prompting)
   - [Price Target Accuracy](#price-target-accuracy)
7. [Rule-Based Judge (--no-llm Mode)](#rule-based-judge----no-llm-mode)
8. [Output Files](#output-files)
9. [ML Model Accuracy](#ml-model-accuracy)
10. [Understanding Verdict Divergence](#understanding-verdict-divergence)
11. [Feedback Loop & Historical Accuracy](#feedback-loop--historical-accuracy)
12. [Disclaimer](#disclaimer)

---

## Quick Start

```bash
# Analyse a single US ticker
python main.py --ticker AAPL

# Skip the LLM step — runs the rule-based judge instead (no API key required)
python main.py --ticker MSFT --no-llm

# Force ML model retraining
python main.py --ticker TSLA --retrain

# Run a batch of tickers from a watchlist file
python main.py --file tickers.txt
python main.py --file tickers.txt --no-llm

# Resolve pending prediction outcomes (fetch actual prices for expired horizons)
python main.py --resolve
python main.py --ticker AAPL --resolve   # resolve one ticker only
python main.py --file tickers.txt --resolve  # resolve all tickers in file
```

Results are printed to the terminal and saved under `reports/{TICKER}_{DATE}/`:
- `report.md` — full Markdown report
- `report.pdf` — PDF version
- `features.csv` — feature matrix used by ML models

---

## Requirements & Installation

Python 3.11+ is required.

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package | Purpose |
|---|---|
| `yfinance` | Free Yahoo Finance data (prices, fundamentals, analyst data, news) |
| `pandas` / `numpy` / `scipy` | Numerical computation |
| `scikit-learn` | Gradient Boosting ML models |
| `matplotlib` | Chart generation |
| `rich` | Colour terminal output |
| `openai` | Azure OpenAI and DeepSeek API client (OpenAI-compatible) |
| `google-genai` | Google Gemini API client |
| `reportlab` | PDF generation |
| `json-repair` | Robust parsing of LLM JSON output |
| `python-dotenv` | `.env` file loading |

---

## Configuration (.env)

Create a `.env` file in the project root. Copy `.env.example` as a starting point.

```ini
# ── LLM Providers (configure at least one) ──────────────────────────
AZURE_OPENAI_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.0-flash

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat          # optional, default: deepseek-chat

# ── Financial constants ──────────────────────────────────────────────
RISK_FREE_RATE=0.0525
EQUITY_RISK_PREMIUM=0.055
MAX_RISK_PER_TRADE=0.01

# ── LLM tuning ───────────────────────────────────────────────────────
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS_AZURE=4000
LLM_MAX_TOKENS_GOOGLE=3000
LLM_MAX_TOKENS_DEEPSEEK=4000
```

If multiple providers are configured, all run independently and their outputs are combined. Each active provider appears in the terminal display and Markdown report. Any provider whose API key is absent is silently skipped — no errors are raised.

---

## How Data Is Collected

All data is fetched free via **Yahoo Finance** using `yfinance`. No paid data API keys are required.

### Step 1 — Price History & Fundamentals (`data/stock_fetcher.py`)

Downloads 2 years of daily OHLCV data. Also fetches ~100 fundamental fields from `yf.Ticker.info`: market cap, sector, P/E, EPS, book value, FCF, revenue growth, ROE, beta, 52-week high/low, dividend yield, etc.

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
| Monte Carlo GBM | 1,000 paths; P10/Median/P90 at 8 horizons |

Monte Carlo drift (μ) blends 60% CAPM-implied return (Rf + β × ERP) and 40% historical mean log-return. Volatility (σ) blends 70% long-term (2-year) and 30% recent (30-day).

### Step 6 — ML Predictions (`ml/trainer.py`, `ml/predictor.py`)

Gradient Boosting trained per ticker on 28 features (returns, MA ratios, oscillators, volume, sentiment). Predicts 5-day and 21-day directional probability + expected return %. Chronological 80/20 split; models cached under `models/{TICKER}/` and auto-retrained after 7 days.

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

`accuracy_pct` is a per-horizon estimated prediction accuracy, derived from the ML classifier test-set accuracy with a decay factor and a VIX penalty. Longer horizons have lower accuracy — this is by design.

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

Each parallel step uses `ThreadPoolExecutor`. The LLM provider callable (`_call_azure` or `_call_google`) is passed as a function argument so the same orchestrator works for both providers.

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

`accuracy_pct` in each target is computed as:

```
base_accuracy  = ML classifier test accuracy for the relevant horizon
                 (clf_5d for ≤1 month, clf_21d for ≥3 months)
decay_factor   = {1W: 1.00, 2W: 0.95, 3W: 0.90, 1M: 0.85,
                  3M: 0.75, 6M: 0.65, 9M: 0.60, 12M: 0.55}
vix_penalty    = −5% for HIGH_FEAR, −2% for ELEVATED, 0% for CALM
accuracy_pct   = clip(base_accuracy × decay_factor + vix_penalty, 30, 85)
```

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

| Signal | Threshold |
|---|---|
| Trailing P/E | <15×: 2pt · 15–25×: 1pt · >40×: 0pt |
| Forward P/E | <15×: 2pt · 15–25×: 1pt · >25×: 0pt |
| PEG Ratio | <1.0: 2pt · 1.0–2.0: 1pt · >2.0: 0pt |
| EV/EBITDA | <10×: 2pt · 10–20×: 1pt · >20×: 0pt |
| DCF vs Price | Undervalued: 2pt · Within 20%: 1pt · Overvalued: 0pt |
| Graham Number vs Price | Below: 2pt · Slightly above: 1pt · Well above: 0pt |
| Price/Book | <1.5×: 2pt · 1.5–3.0×: 1pt · >3.0×: 0pt |
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

Models are trained per ticker on ~2 years of daily data with an 80/20 chronological split.

| Ticker | 5-day Clf | 21-day Clf | 5-day MAE | 21-day MAE |
|---|---|---|---|---|
| AMD | 38% | 50% | 9.3% | 13.0% |
| C6L.SI | 60% | 61% | 1.6% | 2.6% |
| D05.SI | 61% | 66% | 2.4% | 4.3% |
| FIG | 75% | 94% | 5.7% | 5.7% |
| IBM | 47% | 27% | 4.9% | 12.1% |
| IONQ | 42% | 31% | 16.0% | 43.3% |
| MSFT | 58% | 21% | 5.0% | 14.4% |
| ORCL | 50% | 61% | 6.5% | 11.5% |
| PL | 34% | 47% | 12.3% | 29.5% |
| U96.SI | 59% | 59% | 3.1% | 6.9% |
| VT | 45% | 26% | 2.5% | 4.5% |

5-day classification averages ~52% (vs 50% random). 21-day classification degrades significantly — short-horizon features lose predictive power at one-month range. ML predictions are a supplementary input, not a primary signal.

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

2. **Outcome resolution** (`--resolve` flag) — fetches the actual closing price for each expired horizon (1W, 2W, 3W, 1M, 3M, 6M, 9M, 12M) and fills in `actual_price`, `pct_error`, and `dir_correct` (1 if the direction was correct, 0 otherwise).

3. **Bias correction** — once N ≥ 5 resolved rows exist for a horizon, a `correction_factor` is computed:
   ```
   correction_factor = 1 + clamp(mean_bias, -0.20, +0.20)
   ```
   This multiplier is applied to LLM price targets before display and in the Markdown report.

4. **Dynamic lens weights** (rule-based mode) — each lens's (fundamental/technical/valuation/risk) contribution is scaled based on how accurate it has been relative to a 50% baseline, then renormalised. Capped at ±50% of the original weight. Activates at N ≥ 5 resolved rows.

5. **LLM prompt injection** — the Judge agent receives a `[HISTORICAL ACCURACY FOR {TICKER}]` block summarising per-horizon direction accuracy, mean bias, and recommendation hit rate.

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

---

## Disclaimer

FinBot is a research and educational tool. Nothing it produces constitutes financial advice. All outputs — including LLM recommendations, price targets, position sizes, and ML predictions — are generated automatically and may be wrong. Always do your own research before making any investment decision.
