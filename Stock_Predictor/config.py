"""
config.py — Centralised configuration loader for FinBot.

Reads from a .env file (if present) and exposes typed constants
consumed by all other modules. Nothing in this file makes network
calls; it is pure I/O-free configuration.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (silently ignored if absent)
load_dotenv()


# ------------------------------------------------------------------
# Azure OpenAI
# ------------------------------------------------------------------
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# True when both Azure key and endpoint are present.
AZURE_ENABLED: bool = bool(AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT)

# ------------------------------------------------------------------
# Google Gemini
# ------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# True when a Google API key is present.
GOOGLE_ENABLED: bool = bool(GOOGLE_API_KEY)

# ------------------------------------------------------------------
# DeepSeek
# ------------------------------------------------------------------
DEEPSEEK_API_KEY:  str = os.getenv("DEEPSEEK_API_KEY",  "")
DEEPSEEK_MODEL:    str = os.getenv("DEEPSEEK_MODEL",    "deepseek-chat")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# True when a DeepSeek API key is present.
DEEPSEEK_ENABLED: bool = bool(DEEPSEEK_API_KEY)

# ------------------------------------------------------------------
# LLM gate — True when at least one LLM provider is configured.
# ------------------------------------------------------------------
LLM_ENABLED: bool = AZURE_ENABLED or GOOGLE_ENABLED or DEEPSEEK_ENABLED

# ------------------------------------------------------------------
# Financial constants
# ------------------------------------------------------------------
# Risk-free rate used in the Sharpe Ratio formula (annualised).
# Default: 5.25 % — approximate US 3-month T-Bill yield (2024/2025).
RISK_FREE_RATE: float = float(os.getenv("RISK_FREE_RATE", "0.0525"))

# WACC approximation used in the simplified DCF when no company-specific
# WACC is available in the data feed.
DEFAULT_WACC: float = 0.10   # 10 % is a common equity discount rate proxy

# Terminal (perpetual) growth rate for the DCF model.
TERMINAL_GROWTH_RATE: float = 0.03  # 3 % — roughly US long-run GDP growth

# Long-run Equity Risk Premium used in CAPM-implied drift for Monte Carlo.
# Source: Damodaran (2024 update) — geometric average ERP for US equities.
EQUITY_RISK_PREMIUM: float = float(os.getenv("EQUITY_RISK_PREMIUM", "0.055"))  # 5.5 %

# Portfolio risk management — max capital risk per trade (used by Position Sizing agent).
MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))  # 1 %

# ------------------------------------------------------------------
# Data-fetch settings
# ------------------------------------------------------------------
# Length of historical price data fetched from Yahoo Finance.
HISTORY_PERIOD: str = "2y"   # 2 years of daily OHLCV data

# Benchmark ticker used for Beta calculation.
BENCHMARK_TICKER: str = "SPY"

# Number of Monte Carlo simulation paths.
MONTE_CARLO_PATHS: int = 1000

# Prediction horizons in *trading* days (approx 252 per year).
PREDICTION_HORIZONS: dict[str, int] = {
    "1 Week":   5,
    "2 Weeks":  10,
    "3 Weeks":  15,
    "1 Month":  21,
    "3 Months": 63,
    "6 Months": 126,
    "9 Months": 189,
    "1 Year":   252,
}

# ------------------------------------------------------------------
# Output settings
# ------------------------------------------------------------------
# Anchored to the project's own directory so the script works correctly
# regardless of the working directory from which it is invoked.
REPORTS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
CHART_DPI: int = 150               # PNG resolution for saved charts

# ------------------------------------------------------------------
# LLM tuning parameters (overridable via .env)
# ------------------------------------------------------------------
LLM_TEMPERATURE: float       = float(os.getenv("LLM_TEMPERATURE",         "0.0"))
LLM_MAX_TOKENS_AZURE: int    = int(os.getenv("LLM_MAX_TOKENS_AZURE",      "8000"))
LLM_MAX_TOKENS_GOOGLE: int   = int(os.getenv("LLM_MAX_TOKENS_GOOGLE",     "8192"))
LLM_MAX_TOKENS_DEEPSEEK: int = int(os.getenv("LLM_MAX_TOKENS_DEEPSEEK",   "8000"))

# ------------------------------------------------------------------
# Feedback loop settings
# ------------------------------------------------------------------
# Directory where per-ticker feedback CSV files are stored.
FEEDBACK_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")

# Minimum number of resolved predictions needed before bias correction
# and dynamic weight adjustment are activated.
FEEDBACK_MIN_SAMPLES: int = int(os.getenv("FEEDBACK_MIN_SAMPLES", "5"))

# Maximum absolute bias correction applied to LLM price targets (as a fraction).
# 0.20 = ±20% cap. Prevents overcorrection on small samples.
FEEDBACK_MAX_BIAS_CORRECTION: float = float(os.getenv("FEEDBACK_MAX_BIAS_CORRECTION", "0.20"))

# ------------------------------------------------------------------
# ML pipeline settings
# ------------------------------------------------------------------
# Directory where trained model files (.pkl) are stored.
ML_MODELS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Minimum number of labelled rows required before attempting to train.
ML_MIN_TRAIN_ROWS: int = int(os.getenv("ML_MIN_TRAIN_ROWS", "100"))

# Number of days after which persisted models are considered stale and
# will be automatically retrained on the next run.
ML_RETRAIN_DAYS: int = int(os.getenv("ML_RETRAIN_DAYS", "7"))
