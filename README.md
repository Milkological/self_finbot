# FinBot

A Python toolkit for analysing stocks. It combines classic quantitative methods, machine learning and a multi-agent LLM pipeline, and keeps a running record of how accurate its own predictions turn out to be.

> Research and learning project. Not financial advice.

## What's in this repository

| Folder | What it does |
|---|---|
| [`Stock_Predictor/`](Stock_Predictor/README.md) | FinBot, the main tool. Give it a ticker and it produces a full report in the terminal, Markdown and PDF. |
| [`stock_retriever/`](stock_retriever/README.md) | A Reddit scraper that finds which tickers people are discussing and scores the sentiment of each mention. |

## FinBot at a glance

- **Data:** prices, fundamentals, analyst targets, news and options from Yahoo Finance, plus fundamentals from SEC EDGAR.
- **Quant analysis:** technical indicators, fundamental valuation, statistical risk measures and a Monte Carlo simulation.
- **Machine learning:** a GradientBoosting + RandomForest ensemble that predicts 5-day and 21-day direction. It is validated on a chronological train/test split, so the model never sees the future.
- **Two ways to reach a verdict:**
  - **LLM mode:** a multi-agent pipeline (analyst, bull/bear researcher, trading and risk agents, then a judge) across Azure OpenAI, Gemini and DeepSeek.
  - **No-LLM mode:** a deterministic rule-based judge that produces the same report shape with no API keys.
- **Resilience:** per-provider timeouts and a circuit breaker. If an AI provider fails, FinBot falls back to the rule-based judge instead of hanging.
- **Feedback loop:** every prediction is logged, checked against the real price later, and fed back into calibration.

## Quick start

```bash
cd Stock_Predictor
pip install -r requirements.txt
cp .env.example .env              # optional: add LLM keys and SEC contact
python main.py --ticker AAPL --no-llm
```

See [Stock_Predictor/README.md](Stock_Predictor/README.md) for every option and a full explanation of the design.

## Results and limitations

Out-of-sample accuracy from the backfill replay (the newest 20% of history, predicted by models trained only on the older 80%):

| Ticker | Samples | Price target direction, 1 week | Price target direction, 1 month | ML direction, 5 days | ML direction, 21 days |
|---|---|---|---|---|---|
| ILMN | 494 | 55% | 72% | 71% | 74% |
| KEEL | 247 | 52% | 64% | 49% | 41% |
| NOW | 1235 | 44% | 27% | 65% | 57% |
| QUBT | 988 | 44% | 36% | 61% | 67% |

What this shows:

- **There is no consistent edge.** Results swing from well above 50% to well below it depending on the stock. Price-target direction is often worse than a coin flip.
- **The ML numbers look better than they are.** The samples are daily and overlapping, so neighbouring predictions are not independent. A model that simply guesses "up" during an uptrend also scores well.
- **The useful part is the measurement.** FinBot is built to report its own accuracy honestly rather than to claim it can beat the market.

Possible improvements: compare against a buy-and-hold baseline, evaluate on non-overlapping windows, and test across many more tickers.
