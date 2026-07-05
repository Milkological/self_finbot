# Changelog

All notable changes to **FinBot — AI-Powered Stock Analyser** are documented here.  
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

---

## [1.7.0] — 2026-07-06

**Go LLM-free.** Makes the deterministic judge + ML the primary product so the tool
works fully with zero API keys. (The LLM pipeline stays in place, dormant without keys,
for optional use and A/B comparison.)

### Added

#### Offline headline sentiment (`data/sentiment_fetcher.py`)
- `score_headlines()` now falls back to **VADER** (offline lexicon) when no LLM is
  configured — previously headline sentiment was a permanent 0.0 stub in `--no-llm` mode.
  Augmented with a **finance-domain lexicon** (beats/miss/downgrade/plunge/surge/…) so it
  reads financial headlines correctly (bullish set +0.88, was neutral before the lexicon).
  Real sentiment now accumulates daily in `feedback/sentiment_history_*.csv`. Adds
  `vaderSentiment` to requirements.

#### `--no-llm` reports are no longer half-empty (`analysis/rule_based_judge.py`)
- `sentiment_verdict`, `macro_verdict`, and `catalysts` were hardcoded "N/A" stubs; they
  are now derived from data already in hand — VADER sentiment, VIX/vol regime + earnings
  proximity, and rule-based catalysts (next earnings, recent upgrades, analyst upside,
  52-week-high proximity, positive news flow).

#### Cap-aware technical weighting (`analysis/rule_based_judge.py`)
- The backtester found the technical lens predicts on volatile/small names (IC +0.067) but
  **inverts on large caps** (IC −0.039). Above ~$10B market cap the technical lens weight is
  now halved and redistributed to the (cap-agnostic) fundamental & valuation lenses. Surfaced
  as `effective_weights.cap_adjusted`.

#### Training universe (`ml/universe_builder.py`, `--build-universe`)
- One-off resumable batch that builds feature CSVs for ~10 liquid large caps per sector
  (~110 tickers) under `reports/_universe/`, then retrains the pooled global model — taking
  its training pool from ~8 tickers/~7k rows to ~100 tickers/~120k rows (the biggest ML data
  lever). `global_trainer._latest_features_csvs` now pools per-run reports + the universe.

#### Paper-trading scoreboard (`portfolio/simulator.py`, `--portfolio`)
- Replays resolved BUY/STRONG BUY predictions as equal-weight paper trades (entry at call
  price, exit at the resolved actual) and reports win rate, average/conviction-weighted
  return, and excess vs holding SPY over matched windows. The single honest number for "is
  the LLM-free path actually good?" Uses only data the feedback loop already produces;
  `--horizon` and `--provider rule_based` supported.

### Deferred (designed, not built — see ROADMAP)
Composite-threshold validation and VIX-regime-conditional technical scoring both need the
backtester extended to replay the *full composite* (it currently replays single lenses);
CV-fold isotonic calibration, quantile target bands, and a pooled hyperparameter search are
follow-ups best run after the universe build provides the data volume to tune against.

---

## [1.6.0] — 2026-07-06

Roadmap session 1: a diagnostic command and CI/dependency infrastructure.

### Added

#### `--doctor` environment & connectivity health check (`diagnostics.py`)
- One command that answers "why is nothing happening?": probes each configured LLM
  provider with a real fast (≤15s) 1-token ping, Yahoo Finance and SEC EDGAR
  reachability, `lxml` availability (earnings-surprise features), per-ticker ML model
  freshness + the global pooled model, and feedback-CSV schema integrity — printed as a
  pass/warn/fail table. Exits non-zero on any FAIL so it doubles as a pre-flight check.
- On first run it immediately surfaced two real issues: Azure unreachable while
  Google/DeepSeek work (so the pipeline *does* run, just slowly on the reasoner models),
  and 5 legacy feedback CSVs missing the `provider` column (pre-multi-provider drift).
- Backed by a new `ping_providers()` in `analysis/llm_analysis.py` that reuses the
  cached clients and the circuit breaker's connection-error classifier.

#### CI & dependency infrastructure
- `.github/workflows/ci.yml` — runs the (offline, mocked) test suite on Python 3.12/3.13
  for every push/PR touching `Stock_Predictor/`, plus an informational `--doctor` smoke step.
- `requirements.lock` — exact known-good pinned versions (the reportlab/Python-3.14/Pillow-12
  break was a drift bug; this prevents repeats). `requirements.txt` stays ranged for upgrades.
- `.env.example` refreshed with the 1.5.0 resilience keys (`LLM_TIMEOUT`,
  `LLM_MAX_RETRIES`, `LLM_CIRCUIT_THRESHOLD`).

---

## [1.5.0] — 2026-07-05

Hardens the LLM pipeline so a dead provider can no longer freeze a run, fixes a
PDF-generation crash, and adds **point-in-time fundamentals from SEC EDGAR** —
the first leak-free fundamental data in the project.

### Fixed

#### PDF generation crashed on chart images (`output/pdf_generator.py`)
- **What:** `doc.build()` raised `TypeError: cannot unpack non-iterable int object`
  from reportlab's `_py_asciiBase85Encode`. reportlab 4.5 defaults to ASCII85 image
  encoding (`useA85=1`) but its C accelerator isn't compiled here, so it used the
  pure-Python base85 encoder, which is broken on Python 3.14 / Pillow 12.
- **Fix:** force `rl_config.useA85 = 0` (FlateDecode) — bypasses the broken path and
  yields smaller PDFs — plus flatten chart PNGs to RGB (removes the alpha channel,
  a second trigger) before embedding. Verified against real report charts.

#### LLM runs could appear frozen for minutes (`analysis/llm_analysis.py`)
- **What:** provider clients were built with no timeout/retry override, inheriting the
  SDK defaults (600s timeout, 2 retries). A hanging or unreachable provider made every
  one of the 13 agents block in turn, and all errors were swallowed silently — the run
  looked frozen with no explanation.
- **Fix (several parts):**
  - Configurable `LLM_TIMEOUT` (60s) and `LLM_MAX_RETRIES` (1) on all three clients.
  - A per-provider **circuit breaker**: after `LLM_CIRCUIT_THRESHOLD` (2) consecutive
    connection-class failures, that provider's remaining agents skip instantly instead
    of each re-hitting the dead endpoint. A dead provider now costs seconds, not minutes.
  - **Visible errors:** each provider failure is logged and printed (`LLM provider
    unavailable — <name>: <reason>`), no longer silent.
  - **First-class fallback:** when ALL providers fail, `get_llm_analysis` now returns the
    full deterministic rule-based analysis instead of a HOLD/N/A stub.
  - **Live progress:** a callback drives the terminal spinner through each provider/wave
    (`LLM · DeepSeek · Traders + market risk (4/6)`) so a long step always shows motion.

### Added

#### SEC EDGAR point-in-time fundamentals (`data/edgar_data.py`)
- Free, key-less XBRL company-facts API. Quarterly revenue, net income, operating income
  extracted across multiple candidate tags (filers switch tags over time), each figure
  joined onto price rows by its **filing date** — so a trading day only ever sees numbers
  already public. The first genuinely leak-free fundamental data in the project; cached
  7 days under `.cache/edgar/`. Foreign/ADR tickers (not in EDGAR) degrade to NaN.
- **New leak-free ML features** (`data/csv_exporter.py`, added to `FEATURE_COLS` →
  auto-retrain): `edgar_revenue_yoy`, `edgar_ni_yoy` (net-income YoY — split-immune,
  unlike EPS), `edgar_revenue_accel`, `edgar_op_margin`, `edgar_net_margin`, plus
  `earnings_surprise_last` / `earnings_surprise_avg4` from the earnings calendar.
- **Fundamental lens is now backtestable** (`analysis/backtester.py`): `--backtest`
  computes a point-in-time EDGAR fundamental score alongside the technical lens and
  reports its forward-return buckets and rank IC — the fundamental lens could never be
  honestly backtested before (yfinance only exposes current snapshots).
- `lxml` and `pillow` added to `requirements.txt` (earnings-calendar scraping and PDF
  image flattening, respectively).

---

## [1.4.0] — 2026-07-04

This release **fixes real scoring bugs in the rule-based judge**, makes ML validation
**honest (walk-forward CV)**, adds a **pooled cross-ticker model**, a **shared data cache**
that removes every duplicate network fetch, and a new **`--discover` mode** that scans the
whole market for fresh candidate tickers.

### Fixed

#### Rule-based judge crashed on strongly bearish stocks (`analysis/rule_based_judge.py`)
- **What:** the STRONG BEARISH TREND label used an invalid f-string format spec
  (`{pdi:.1f if pdi else '?'}`), raising `ValueError` whenever ADX > 25 with −DI dominant.
  The exception was swallowed in `main.py`, so `--no-llm` silently returned the
  "Rule-based analysis failed" stub for exactly the stocks that most needed a bearish verdict.
- **Improvement:** values are pre-formatted; bearish names now score normally.

#### Small ML adjustments were silently dropped (`analysis/rule_based_judge.py`)
- **What:** when the Monte Carlo nudge recomputed the composite, the ML adjustment was only
  re-applied when its note string was non-empty — i.e. adjustments under 1 pt vanished.
- **Improvement:** the adjustment is computed once and always re-applied.

#### Degenerate sentiment feature removed from ML (`ml/trainer.py`, `feedback/tracker.py`)
- **What:** `sentiment_score` was 0.0 on every historical training row and only nonzero on the
  inference row — the model learned a constant, then saw an out-of-distribution value live.
- **Improvement:** dropped from `FEATURE_COLS`; the retrain-detection check was also tightened
  from subset to equality so *removing* a feature now triggers retraining. Real per-day
  sentiment is now archived to `feedback/sentiment_history_{TICKER}.csv` on every run so it can
  return as a leak-free feature once enough history accumulates.

#### Backfill stays out-of-sample under the new trainer (`feedback/backfiller.py`)
- **What:** persisted models are now fitted on *all* rows (see below), so replaying them over
  the last 20% would have been in-sample and inflated bootstrap accuracy.
- **Improvement:** the backfiller fits its own temporary models on the first 80% only.

### Changed

#### Honest walk-forward validation (`ml/trainer.py`)
- Single 80/20 split → **5-fold expanding-window CV** (`TimeSeriesSplit`) with a **gap equal to
  the label horizon** (the old split leaked: forward-looking labels of the last train rows
  overlapped the test window). Reported accuracy is the fold mean; std, per-fold values, and a
  **LogisticRegression baseline** (overfit detector) are new additive metrics. Final persisted
  models now train on **all** labelled rows instead of discarding the newest 20%.
- **Expect reported accuracies to drop toward ~50–58%.** That is the honest number — the old
  one was optimistic.

#### Accuracy-gated ML modifier (`analysis/rule_based_judge.py`)
- The ±10-pt ML composite adjustment is now scaled per horizon by validated CV skill:
  ≤50% accuracy → zero influence; ≥65% → full weight. Calibrated probabilities are preferred.
  A coin-flip model can no longer swing a recommendation.

#### Shared market-data cache (`data/market_data.py`, new)
- One `yf.Ticker`, one `.info`, one VIX series, one SPY history, one earnings calendar
  **per process** (previously: `.info` ×3, VIX ×2, earnings ×2 per run). Batch mode prewarms
  all tickers through a small thread pool before the sequential pipeline, and tests now mock a
  single choke point.

### Added

- **New leak-free ML features** (`data/csv_exporter.py`): relative strength vs SPY (21/63d),
  overnight gap + 5-day mean, normalised ATR, drawdown-from-peak, 20-day volume z-score.
- **Pooled cross-ticker model** (`ml/global_trainer.py`, `--retrain-global`): one model trained
  on every ticker's stacked history; the predictor blends it with per-ticker models weighted by
  CV accuracy and uses it alone for tickers too young to train — the accuracy lever for
  newly discovered names.
- **Tuned decision thresholds** (`feedback/accuracy.py`): with ≥30 resolved outcomes, the
  probability cutoff that maximises balanced accuracy is stored in `calibration.json` and used
  by the predictor instead of a hard-coded 0.5.
- **`--discover` mode** (`discovery/`): scans yfinance predefined screeners, a full-universe
  volume-spike scan, 52-week-high breakouts, and a NASDAQ-directory new-listing diff; filters
  for tradability; ranks candidates with a quick two-lens rule-based score; appends the best to
  `watchlist.txt` with provenance (never touches `tickers.txt`). Ranked CSV under
  `reports/discovery/`.
- Watchlist/ticker files now support **inline `#` comments**.
- Support/resistance pivot detection vectorised (was an O(n·window) Python loop).
- **Backtest harness** (`analysis/backtester.py`, `--backtest` / `--backtest-broad`):
  replays the technical lens weekly across full price history (leak-free — indicators are
  past-only rolling computations; 52-week stats reconstructed from trailing 252-bar
  extremes) and reports forward returns by score bucket, hit rates, Spearman rank IC,
  and excess return vs SPY, with a full-vs-thin signal split. First run on the live
  watchlist showed the ≥70 bucket earning a 60–64% hit rate vs ~45–49% for lower buckets.
  Fundamental/valuation lenses are deliberately NOT replayed (current-snapshot data would
  be look-ahead bias) and every run prints its survivorship/overlap caveats.
- **Sector-relative valuation** (`data/sector_data.py`): trailing P/E, forward P/E and P/B
  are now graded against the stock's own sector-peer medians (top ~100 US names by market
  cap, one key-free Yahoo screener request per sector, cached 7 days, with stale-cache and
  static-table fallbacks). A 30× P/E is cheap for semis and expensive for banks — absolute
  bands systematically punished growth sectors and flattered deep value. Signals now read
  e.g. "DISCOUNT to sector (21.1× vs peer median 26.5×)". PEG/EV-EBITDA/P-S/DCF/Graham keep
  their absolute thresholds (peer medians for those aren't available from screener quotes).

---

## [1.3.0] — 2026-06-24

This release **repairs and grounds the feedback loop**, **speeds up the LLM pipeline**, and
makes the **terminal display and report far easier to read**. Each item below states *what*
changed, *why*, and *how it improves things*.

### Fixed

#### Feedback resolver was silently dead on modern yfinance (`feedback/resolver.py`)
- **What:** `_fetch_price_on_or_before()` called `yf.download(..., show_errors=False)`. That
  keyword was removed from current yfinance, so the call raised `TypeError`, which the bare
  `except` swallowed — the function always returned `None`.
- **Why it mattered:** *no live prediction ever resolved*, so accuracy stats, bias correction,
  dynamic lens weights, and ML calibration never activated. The whole learning loop was inert.
- **Improvement:** kwarg removed; the loop accumulates real outcomes again. Verified live —
  a single `--resolve KEEL` settled **247** previously-stuck outcomes.

#### Resolver: batched, pandas-3-safe rewrite (`feedback/resolver.py`)
- **What:** resolution now downloads each ticker's price history **once** and resolves every
  row/horizon from that in-memory series (was one `yf.download` *per horizon per row* — up to
  ~2,000 calls for a 250-row CSV). CSV is read as object dtype (`keep_default_na=False` +
  `astype(object)`) so outcome writes no longer hit pandas ≥ 3's strict StringDtype.
- **Improvement:** ~100× fewer network calls per resolve pass; resolution works on pandas ≥ 3.

#### Calibration no longer dies on imperfect CSVs (`feedback/accuracy.py`)
- **What:** `build_calibration_table()` used a plain `read_csv` that threw on CSVs containing a
  single malformed row (the "Expected 72 fields, saw 73" corruption seen in some feedback files).
- **Improvement:** reads with `on_bad_lines="skip"` like the rest of the module, so calibration
  builds for **every** ticker, not just pristine ones.

#### Data fetch robustness (`data/stock_fetcher.py`, `analysis/statistical.py`)
- **What:** removed the same removed-kwarg `show_errors=False` from `fetch_vix_data()` (it was
  silently zeroing the VIX feature); `_get_spy_data()` now flattens MultiIndex columns.
- **Improvement:** the VIX features populate, and Beta no longer silently fails on yfinance
  versions that return MultiIndex columns for a single symbol.

#### Dead error-branch in the LLM orchestrator (`analysis/llm_analysis.py`)
- **What:** `get_llm_analysis()` checked `recommendation == "ERROR"`, but `_error_result()`
  returns `"HOLD"` — so the failed-provider summary was always empty and bias correction could
  be applied to failed providers. Now keys off the authoritative `llm_available` flag.

#### Test isolation + platform fixes
- **`tests/test_feedback_tracker.py`:** removed an `importlib.reload()` that re-ran
  `from config import FEEDBACK_DIR` and defeated the `patch(...)`, causing tests to write to the
  **real** `feedback/` directory and depend on accumulated rows. Tests are now hermetic.
- **`main.py`:** forces UTF-8 on stdout/stderr at startup — fixes `UnicodeEncodeError` for
  symbols like σ/β when the console or a pipe defaults to cp1252 on Windows.
- **Repo hygiene:** untracked 18 `.pyc` files that had been committed before `.gitignore` existed.

### Changed

#### LLM pipeline: concurrent providers + reused clients (`analysis/llm_analysis.py`)
- **What:** the Azure / Gemini / DeepSeek chains now run **concurrently** in a
  `ThreadPoolExecutor` (each chain still parallelises its own 13 agents). Each provider's API
  client is constructed **once** via `lru_cache` (`_get_azure_client` / `_get_google_client` /
  `_get_deepseek_client`) instead of on every agent call.
- **Why:** the chains are independent and the SDK clients are thread-safe and pool connections.
- **Improvement:** wall-clock ≈ the *slowest single provider* instead of the *sum* of all
  providers; far fewer redundant client constructions per run.

#### Bias correction: median + sample shrinkage (`feedback/accuracy.py`)
- **What:** per-horizon bias now uses the **median** signed error, and the correction is
  **shrunk** toward 1.0 on thin data:
  `1 + clamp(median_bias/100, ±FEEDBACK_MAX_BIAS_CORRECTION) × min(1, N / (2·FEEDBACK_MIN_SAMPLES))`.
- **Why:** the mean let one volatile-horizon outlier swing every future target; small samples
  over-corrected.
- **Improvement:** corrections are robust to outliers and ramp in gradually as evidence grows.

#### The Judge is grounded in *realized* accuracy (`feedback/accuracy.py`, `analysis/llm_analysis.py`)
- **What:** `format_llm_injection()` now instructs the Judge to use each horizon's **realized
  directional accuracy** as the base for its `accuracy_pct` when N ≥ `FEEDBACK_MIN_SAMPLES`
  (falling back to the ML-test-split × decay × VIX heuristic only when there is no track record),
  and adds caution rules: rec-hit-rate < 50% caps confidence at MEDIUM, < 40% caps at LOW, and
  sub-coin-flip horizons are flagged. `_build_judge_prompt()`'s ACCURACY_PCT block was updated to
  match.
- **Improvement:** stated confidence/accuracy reflect how FinBot has *actually* performed and
  self-correct as resolved data accumulates, instead of being persistently over-confident.

#### Display & report readability (`output/terminal_display.py`, `output/report_generator.py`)
- **What:** the terminal now opens with a bottom-line-up-front **Executive Summary** panel
  (recommendation, confidence, price → key target, ST/LT outlook, ML cross-check, top bull/risk),
  and every analysis domain has its **own accent colour** via shared `_section()` / `_std_table()`
  helpers. The Markdown report leads with an **Executive Summary** table and a **Contents** TOC.
- **Improvement:** the verdict is the first thing you see, and sections are distinguishable at a
  glance instead of a uniform cyan/magenta wall.

### Added

#### Tests (`tests/test_resolver.py`, `tests/test_accuracy.py`)
- First unit coverage for the resolver and accuracy modules: single batched fetch, no
  `show_errors` kwarg, outcome/directional math, unelapsed-horizon handling, median+shrinkage
  correction math, calibration robustness to malformed rows, and the grounded/caution injection.
- Suite grew from 176 → **191 tests**, all green.

---

## [1.2.0] — 2026-06-22

### Added

#### Feedback Loop — Historical Backfill (`feedback/backfiller.py`)
- New module `feedback/backfiller.py` solving the feedback cold-start problem.
- `backfill_historical_predictions(ticker, trainer_result, features_csv, price_df)` — replays trained models across the **test split** (last 20% of labelled rows in `features.csv`) to immediately generate resolved feedback rows without waiting for real runs to accumulate.  Only out-of-sample rows are used to avoid measuring in-sample performance.
- For each test-split row on date D the function: builds the feature vector, runs all 4 GBM + 4 RF models, looks ahead in `price_df` to resolve actual prices at all 8 horizons (1W=5 biz days … 12M=252 biz days), computes `dir_correct_{h}`, `ml_5d_correct`, and `ml_21d_correct`, then appends a fully-resolved row tagged `mode="backfill"`, `provider="backfill"` to the feedback CSV.
- `needs_backfill(ticker)` — returns True when the feedback CSV has fewer resolved rows than `FEEDBACK_MIN_SAMPLES`. Used as the auto-trigger gate inside `run_pipeline()`.
- Deduplication by `run_date` prevents double-writing if `--backfill` is run more than once.
- **What backfill provides:** ML accuracy calibration, all-horizon directional accuracy stats, adjusted lens weights (N≥5). **What it cannot provide:** LLM price target bias (`pct_error`) — no historical LLM outputs exist, so `correction_factor` still requires real LLM runs.

#### Feedback Loop — ML Probability Calibration (`feedback/accuracy.py`)
- `build_calibration_table(ticker)` — buckets resolved `ml_Xd_prob_up` values into ranges `[0.0–0.3, 0.3–0.4, …, 0.8–1.0]`, computes empirical direction-accuracy per bucket, and saves the result to `models/{TICKER}/calibration.json`. Activated when resolved rows ≥ `CALIBRATION_MIN_SAMPLES` (default 10). Called automatically after backfill and after each pipeline run that triggers backfill.

#### ML — GradientBoosting + RandomForest Ensemble (`ml/trainer.py`, `ml/predictor.py`)
- `ml/trainer.py`: each horizon (5d, 21d) now trains **two** classifier/regressor pairs — `GradientBoostingClassifier` (existing, primary) and a new `RandomForestClassifier` companion. Same for regressors. RF hyper-parameters: `n_estimators=200`, `max_depth=5`, `min_samples_leaf=5`, `max_features="sqrt"`. Models saved as `clf_{h}d_rf.pkl` and `reg_{h}d_rf.pkl`.
- `ml/predictor.py`: ensemble inference — raw `predict_proba` probabilities from GBM and RF are averaged; the averaged value is used as `clf_{h}d_prob_up`. Ensemble averaging reduces variance and gives more stable signals.
- `ml/predictor.py`: **calibrated probability** — after raw ensemble probability is computed, `_apply_calibration()` looks up `models/{TICKER}/calibration.json`; if the bucket exists the empirical accuracy replaces the raw prob as `clf_{h}d_prob_up_cal`. Falls back to raw prob when calibration is unavailable.
- `ml/predictor.py`: **confidence labels** — `clf_{h}d_confidence` set to `"HIGH"` when `|calibrated_prob − 0.5| > 0.20`, `"MEDIUM"` when `> 0.10`, `"LOW"` otherwise.
- `ml/trainer.py`: **feature importances** — top-10 GBM `feature_importances_` (sorted descending) saved to `metrics.json` as `top_features_clf_5d`, `top_features_clf_21d`, `top_features_reg_5d`, `top_features_reg_21d`.
- `_load_models()` updated to load RF companions when present; silently skips them when absent (backward compatibility with existing `*.pkl` files from v1.1).

#### ML — 4 New Features (`data/csv_exporter.py`, `ml/trainer.py`)
- **`vix_level`** — daily VIX close joined to the price DataFrame by date. Provides macro fear-regime context missing from all previous feature sets. Source: `^VIX` via yfinance (fully historical, no limitations).
- **`vix_5d_change`** — 5-day percentage change in VIX; captures whether fear is rising or falling at prediction time.
- **`earnings_within_14d`** — binary flag (1/0); 1 if the row's date is within 14 calendar days of any earnings announcement. Derived from `yf.Ticker.get_earnings_dates(limit=20)`. Earnings proximity is one of the strongest disruptors of technical patterns.
- **`price_zscore_20d`** — `(Close − SMA_20) / rolling_std_20`. A mean-reversion signal orthogonal to Bollinger %B (which normalises by bandwidth rather than standard deviation). Pure OHLCV computation; no new data source required.
- All 4 features added to `FEATURE_COLS` in `ml/trainer.py`, bringing the total feature count from 29 → 33.

#### Data — VIX Fetch (`data/stock_fetcher.py`)
- `fetch_vix_data(start, end)` — downloads daily `^VIX` close prices for a given date range via yfinance. Returns a tz-naive `pd.Series` indexed by date. Returns an empty Series on any failure so callers degrade gracefully.

#### CLI — `--backfill` Flag (`main.py`)
- New `--backfill` argument — resolves historically by running the trained model over test-split data and writing resolved feedback rows, then exits. Accepts `--ticker`, `--file`, or falls back to `tickers.txt`.
- `run_pipeline()` auto-triggers backfill (once) when `needs_backfill(ticker)` is True after ML training, then calls `build_calibration_table()` to immediately activate calibration.

#### Config — New Keys (`config.py`)
- `BACKFILL_MIN_TEST_ROWS` (default `20`, env: `BACKFILL_MIN_TEST_ROWS`) — minimum test-split rows required before backfill writes rows. Prevents writing noise on very thin datasets.
- `CALIBRATION_MIN_SAMPLES` (default `10`, env: `CALIBRATION_MIN_SAMPLES`) — minimum resolved rows before the ML probability calibration table is built.

### Changed

#### Data — Extended Training History (`config.py`, `data/stock_fetcher.py`)
- `HISTORY_PERIOD` changed from `"2y"` to `"5y"` — approximately 2.5× more training data (~1,250 daily rows vs ~500). This is the single highest-ROI change for ML accuracy: more data improves generalisation, reduces overfitting on short-period patterns, and allows the test split to provide ~250 genuinely out-of-sample rows for backfill.
- All downstream consumers (csv_exporter, ML trainer, statistical models) automatically benefit; no other code changes required.

#### Analysis — Monte Carlo Simulation (`config.py`)
- `MONTE_CARLO_PATHS` increased from `1000` to `2500`. Tighter confidence intervals at 5th/95th percentiles with minimal additional runtime (~150ms on modern hardware).

#### LLM — Error Result No Longer Returns "ERROR" Recommendation (`analysis/llm_analysis.py`)
- `_error_result()` now returns `"HOLD"` instead of `"ERROR"` as the `recommendation` field. The authoritative signal for a failed LLM run is `llm_available=False`, not the recommendation string.
- **Why this matters:** the feedback tracker previously skipped saving rows where `recommendation == "ERROR"`, which silently discarded every failed LLM run from the feedback loop. With `"HOLD"`, these runs are now saved and resolved correctly — directional accuracy for the ML signals and rule-based fallback is captured even when the LLM chain fails.
- `main.py` feedback-save guard updated to check `presult.get("llm_available", True) == False` instead of `recommendation == "ERROR"`.

#### CSV Exporter — Signature Updated (`data/csv_exporter.py`)
- `export_features_csv()` and `_build_feature_df()` now accept `ticker: str` as a first parameter (required for earnings-date lookup). All call sites in `main.py` updated accordingly.
- Forward-return label section renumbered (4 new feature sections inserted before it).

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
