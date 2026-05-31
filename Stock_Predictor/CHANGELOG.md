# Changelog

All notable changes to **FinBot — AI-Powered Stock Analyser** are documented here.  
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

---

## [1.1.0] — 2026-05-31

### Added

#### Feedback Loop & Historical Accuracy (`feedback/` package)
- `feedback/__init__.py` — Package init for the feedback loop submodules.
- `feedback/tracker.py` — `save_prediction()` appends one row to `feedback/{TICKER}_feedback.csv` after each pipeline run, capturing ticker, date, mode, current price, all LLM/rule-based price targets (8 horizons), ML signals, analyst data, options metrics (put/call ratio, average IV), short interest, volume, and news headlines.
- `feedback/resolver.py` — `resolve_pending(ticker)` fetches actual closing prices for every expired horizon row, computing `actual_price`, `pct_error`, and `dir_correct`. `resolve_all(tickers)` resolves a batch. Triggered via the new `--resolve` CLI flag.
- `feedback/accuracy.py` — `get_context(ticker, mode)` computes per-horizon direction accuracy, mean bias, and `correction_factor = 1 + clamp(mean_bias, ±0.20)`. Also computes ML accuracy, recommendation hit rate, and dynamically adjusted rule-based lens weights. Returns a context dict consumed by the analysis and display layers. `apply_bias_correction(targets, ctx)` multiplies LLM price targets by correction factors. `format_llm_injection(ticker, ctx)` formats a `[HISTORICAL ACCURACY]` block injected into the Judge prompt.

#### CLI — `main.py`
- `--resolve` flag — resolves pending prediction outcomes for known tickers (from `--ticker`, `--file`, or `tickers.txt` default) and exits without running full analysis.
- Imports and wires the full feedback loop: `resolve_pending` called at pipeline start, `get_accuracy_context` computed, `save_feedback` called after `display_full_report`.
- `fetch_options_data(ticker)` called after analyst data fetch (Step 2); result passed to `save_feedback`.
- `accuracy_context` passed to `run_rule_based_analysis`, `get_llm_analysis`, `run_rule_based_judge`, `display_full_report`, and `generate_markdown_report`.

#### Data — `data/stock_fetcher.py`
- `fetch_options_data(ticker)` — fetches nearest-expiry options chain via yfinance, computes `put_call_ratio` and `options_iv_avg`; returns `{...: None}` gracefully on any error.

#### Analysis — `analysis/llm_analysis.py`
- `get_llm_analysis()` accepts `accuracy_context: dict = None`; passes it through to `_run_agent_chain` and `_build_judge_prompt`.
- `_build_judge_prompt()` appends a `[HISTORICAL ACCURACY]` block to the Judge prompt via `_get_accuracy_injection()` when context has data.
- `_get_accuracy_injection()` helper — lazy-imports `format_llm_injection` from the feedback package to avoid circular imports.
- After the Judge returns, `apply_bias_correction()` is applied to `target_prices` when accuracy context is available.

#### Analysis — `analysis/rule_based_judge.py`
- `run_rule_based_judge()` and `run_rule_based_analysis()` accept `accuracy_context: dict = None`.
- Dynamically substitutes adjusted lens weights from the feedback loop when `weights_adjusted=True` and N ≥ 5 resolved rows exist.
- Return dict now includes `effective_weights` showing the weights actually used.

#### Output — `output/terminal_display.py`
- `display_full_report()` accepts `accuracy_context: dict = None`.
- New `display_historical_accuracy(accuracy_context)` function — renders a Rich panel with a per-horizon table (Horizon | N | Dir Accuracy % | Mean Bias | Correction), ML accuracy, recommendation hit rate, and dynamically adjusted weights if applicable.

#### Output — `output/report_generator.py`
- `generate_markdown_report()` accepts `accuracy_context: dict = None`.
- New `## Historical Accuracy (Feedback Loop)` Markdown section inserted after the ML Predictions block; includes a per-horizon table with direction accuracy, mean bias, and correction factor.

#### Config — `config.py`
- `FEEDBACK_DIR` — absolute path to the `feedback/` directory; overridable via env.
- `FEEDBACK_MIN_SAMPLES` — minimum resolved rows before bias correction and weight adjustment activate (default `5`, env: `FEEDBACK_MIN_SAMPLES`).
- `FEEDBACK_MAX_BIAS_CORRECTION` — maximum bias correction magnitude, clamped ±N (default `0.20`, env: `FEEDBACK_MAX_BIAS_CORRECTION`).

#### DeepSeek API Support
- `config.py` — Added `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` (default `deepseek-chat`), `DEEPSEEK_BASE_URL`, `DEEPSEEK_ENABLED`, and `LLM_MAX_TOKENS_DEEPSEEK`. Updated `LLM_ENABLED` gate to include DeepSeek.
- `analysis/llm_analysis.py` — Added `_call_deepseek()` function using the OpenAI-compatible client (`openai.OpenAI` with custom `base_url`). DeepSeek runs as a third provider alongside Azure and Google; silently skipped when `DEEPSEEK_API_KEY` is absent.

#### Per-Provider Feedback Tracking
- `feedback/tracker.py` — Added `provider` column to `COLUMNS`; `save_prediction()` now accepts a `provider` parameter. In LLM mode, one row is saved per active provider per run (e.g., "Azure OpenAI", "DeepSeek (deepseek-chat)"). In no-LLM mode, `provider="rule_based"`.
- `feedback/accuracy.py` — `get_context()` now accepts `provider: str | None = None`; filters the feedback CSV by provider when specified (backward-compatible with existing CSVs that lack the column).
- `analysis/llm_analysis.py` — `get_llm_analysis()` accepts `accuracy_context_map: dict | None` so each provider chain receives its own historical accuracy context for bias correction and prompt injection.
- `main.py` — Builds a per-provider `accuracy_context_map` at pipeline start; passes it to `get_llm_analysis`; iterates `llm_result["providers"]` to save one feedback row per provider.

#### Display — active providers shown
- `output/terminal_display.py` — `display_full_report()` prints an "Analysis providers" line listing all active (non-error) providers that contributed to the run.
- `output/report_generator.py` — `generate_markdown_report()` writes a `**Providers used:**` line under `## AI Analyst Recommendation`.
- `README.md` — Updated Configuration section to document DeepSeek env vars; updated Output Files table to reflect one row per provider per run.

#### Tests
- `tests/test_feedback_tracker.py` — Unit tests for `feedback/tracker.py`: verifies CSV creation with correct header on first call, presence of all `COLUMNS` as headers, correct recording of the `provider` field, row-append behaviour on subsequent calls, valid returned file path, and graceful handling of minimal inputs (empty `llm_result`, empty `ml_result`).

---

## [1.0.0] — 2026-05-31

Initial public release of the full FinBot analysis pipeline.

### Added

#### Project infrastructure
- `config.py` — Centralised, type-annotated configuration loaded from `.env`.  
  Exposes Azure OpenAI, Google Gemini, financial constants (WACC, risk-free rate,
  CAPM ERP), and ML settings. Pure I/O-free module (no network calls at import time).
- `requirements.txt` — Pinned minimum versions for all runtime dependencies.
- `.gitignore` — Excludes virtualenv, generated reports, trained model binaries,
  credentials (`.env`), bytecode, and IDE artefacts.
- `tickers.txt` — Example watchlist file used with the `--file` batch-run flag.

#### CLI — `main.py`
- `--ticker` / `-t` — Analyse a single Yahoo Finance ticker symbol.
- `--file` / `-f` — Read a watchlist `.txt` file and run the full pipeline for
  every ticker listed, continuing after per-ticker failures.
- `--no-llm` — Skip LLM agents entirely; falls back to the rule-based judge.
- `--retrain` — Force retraining of ML models even when cached models are fresh.
- Rich progress bar (spinner + elapsed time) for every pipeline step.

#### Data layer
- `data/stock_fetcher.py` — Downloads 2-year daily OHLCV history (auto-adjusted
  for splits/dividends) and 100+ fundamental fields via `yfinance`. Appends a
  `Log_Return` column. Raises `ValueError` for invalid / delisted tickers.
- `data/analyst_fetcher.py` — Fetches analyst consensus key, price targets
  (mean / low / high), recommendation history, upcoming earnings dates, and
  up to 10 recent news headlines from Yahoo Finance.
- `data/sentiment_fetcher.py` — Scores a batch of news headlines on a
  −1.0 → +1.0 scale using Azure OpenAI or Google Gemini (tried in that order).
  Returns a neutral stub when no LLM provider is configured or every call fails.
- `data/csv_exporter.py` — Builds a 30+ column feature matrix per trading day
  (technical indicator ratios, rolling aggregates, lagged log-returns, volume
  ratio, sentiment score) and writes it to `features.csv` for ML use.

#### Analysis layer
- `analysis/technical.py` — Ten native technical indicators with academic
  source citations and trading rationales:
  - SMA (20 / 50 / 200 periods)
  - EMA (12 / 26 periods)
  - RSI-14 (Wilder smoothing)
  - MACD (12 / 26 / 9)
  - Bollinger Bands (20 periods, 2σ) + %B + bandwidth
  - Fibonacci retracement levels (0 % → 100 %)
  - Local support / resistance pivots (rolling-window swing highs/lows)
  - ATR-14 (Wilder True Range)
  - ADX-14 with +DI / −DI directional indicators
  - On-Balance Volume (OBV)
- `analysis/fundamental.py` — Five valuation metrics:
  - Trailing and Forward P/E ratios with tiered assessments
  - PEG Ratio (Lynch growth-adjusted P/E)
  - Graham Number (√(22.5 × EPS × Book Value))
  - Simplified DCF with 2-stage terminal value (10 % WACC, 3 % perpetuity growth)
  - Price-to-Book ratio and Dividend Yield
- `analysis/statistical.py` — Seven statistical models:
  - Historical (Realised) Volatility — annualised daily σ + regime detection
  - Beta vs SPY (1-year OLS covariance)
  - Sharpe Ratio (excess return / σ, 5.25 % risk-free rate)
  - Sortino Ratio (downside-only σ denominator)
  - Maximum Drawdown and Calmar Ratio (CAGR / MDD)
  - OLS linear regression on log-prices — trend-based price extrapolation
  - Monte Carlo GBM — 1,000 paths, CAPM-blended drift (60 % CAPM / 40 %
    historical), P10 / median / P90 predictions at 8 horizons
- `analysis/llm_analysis.py` — Multi-agent LLM pipeline using ReAct prompting
  (Thought → Action → Observation → Answer). Five teams, 13 agents:
  - **Team 1 — Analyst**: Fundamental, Sentiment, News, Technical agents (4, parallel)
  - **Team 2 — Researcher**: Bullish, Bearish, Synthesiser agents (3, parallel then serial)
  - **Team 3 — Trading**: Momentum, Value, Swing Trader agents (3, parallel)
  - **Team 4 — Risk Management**: Market Risk, Portfolio Risk agents (2, parallel)
  - **The Judge**: Synthesises all four teams + ML signal → final recommendation,
    8-horizon price targets, entry / stop / exit levels, position sizing
  - Supports Azure OpenAI and Google Gemini; degrades gracefully to rule-based
    analysis when no provider is configured
- `analysis/rule_based_judge.py` — Deterministic 4-lens investment judge (no LLM):
  - Fundamental lens (30 %): ROE, margins, FCF yield, interest coverage, D/E
  - Technical lens (25 %): RSI, MACD, price vs SMAs, ADX, Bollinger %B,
    52-week position, OBV, linear-regression trend
  - Valuation lens (30 %): P/E, PEG, DCF vs price, Graham Number, P/B,
    analyst consensus target
  - Risk lens (15 %): Sortino, Calmar, Beta, volatility regime, earnings proximity
  - Composite score 0–100 → STRONG BUY / BUY / HOLD / SELL / STRONG SELL
  - ML signal modifier (bounded ±10 pts) and Monte Carlo valuation nudge
  - Hard veto guards for distressed situations (negative equity, extreme volatility)

#### Machine-learning layer
- `ml/trainer.py` — Trains four GradientBoosting models per ticker:
  - `clf_5d` / `clf_21d` — direction classifiers (GradientBoostingClassifier)
  - `reg_5d` / `reg_21d` — return regressors (GradientBoostingRegressor)
  - Chronological 80/20 time-series split (leakage-free)
  - Auto-retrains when persisted models are older than `ML_RETRAIN_DAYS`
  - Persists `.pkl` model files, `features.json`, and `metrics.json` under
    `models/{TICKER}/`
- `ml/predictor.py` — Loads the latest feature row from `features.csv` and
  runs inference; returns probability-of-up (5d / 21d), predicted % return,
  direction string, and training-time accuracy / MAE metrics.

#### Output layer
- `output/terminal_display.py` — Rich terminal UI: colour-coded signal tables
  (green = bullish, red = bearish, yellow = neutral), per-agent panels for all
  13 LLM agents, rule-based judge panel, ML prediction panel, banner warnings
  when LLM is unavailable.
- `output/report_generator.py` — Saves Markdown report and three PNG charts:
  - Price + SMA-20/50/200 + Bollinger Bands + volume subplot
  - Technical subplots (RSI, MACD, price)
  - Monte Carlo fan chart (P10 / median / P90, 1,000 paths)
- `output/pdf_generator.py` — Generates a 25+ page professional PDF report:
  - Cover page with recommendation badge and company overview
  - Executive summary (LLM investment thesis + team summaries)
  - Technical Analysis section (7 indicators with formula, source, rationale)
  - Fundamental Analysis section (5 metrics with DCF model detail)
  - Statistical Analysis section (5 models with derivation + predictions table)
  - Analyst consensus history and legend
  - All charts embedded with annotations
  - Mathematical appendix
  - Regulatory disclaimer

#### Tests
- `tests/test_config.py` — Config constant types and sensible value ranges.
- `tests/test_technical.py` — All 10 technical indicator functions (pure, no network).
- `tests/test_fundamental.py` — All 5 fundamental valuation functions (pure).
- `tests/test_statistical.py` — All 7 statistical models; beta mocked via SPY patch.
- `tests/test_rule_based_judge.py` — Rule-based judge helpers and composite scoring.
- `tests/test_csv_exporter.py` — Feature CSV creation with temp directory.
- `tests/test_stock_fetcher.py` — Stock data fetcher with mocked yfinance.
- `tests/test_analyst_fetcher.py` — Analyst data fetcher with mocked yfinance.
- `tests/test_sentiment_fetcher.py` — Headline scorer with mocked LLM calls.
- `tests/test_ml_trainer.py` — ML trainer with synthetic CSV data.
- `tests/test_ml_predictor.py` — ML predictor with synthetic trained models.
