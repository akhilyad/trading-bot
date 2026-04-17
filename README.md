<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=ELITE+QUANT+BOT+v5.5&fontSize=80&animation=waving&fontAlign=50" />
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python" /></a>
  <a href="https://pypi.org/project/kiteconnect/"><img src="https://img.shields.io/badge/Kite%20Connect-API-green?style=flat&logo=zerodha" /></a>
  <a href="https://telegram.org"><img src="https://img.shields.io/badge/Telegram-Notifications-blue?style=flat&logo=telegram" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat" /></a>
  <img src="https://img.shields.io/badge/Status-ACTIVE🚀-red?style=flat" />
</p>

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=wave&color=gradient&height=80&section=header" />
</p>

# ⚡ ELITE QUANT BOT v5.5 - ULTIMATE GOD MODE TRADING SYSTEM ⚡

> The most advanced, fully-integrated algorithmic trading bot for Indian markets (NSE/BSE). Self-learning. Self-evolving. Unbeatable.

---

## 🚀 Features (ALL 25+ Integrated)

| Category | Features |
|----------|----------|
| 🤖 **AI/ML** | Deep RL (DQN), Genetic Algorithm, Bayesian/GP, HMM Regime Detection |
| 📰 **Data** | Multi-timeframe (20+ indicators), News Sentiment, Social Listening |
| 💰 **Trading** | Multi-leg Options, Execution Algos (VWAP/TWAP/IS/Iceberg) |
| 🛡️ **Risk** | VaR/CVaR, Circuit Breakers (4-level), Emergency Liquidation |
| 🔮 **Analysis** | Volatility Surface, IV Rank, Calendar Anomalies, Pairs Trading |
| 📊 **Research** | Forward Testing, XAI (SHAP), Multi-Agent Coordination |

---

## 🏃‍♂️ Quick Start

```bash
# 1. Clone
git clone https://github.com/akhilyad/trading-bot.git
cd trading-bot

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Generate token
python setup.py

# 5. Run (paper mode)
python elite_quant_bot_v5.py
```

> ⚠️ **ALWAYS start with paper trading!** Test for 1-2 months before live trading.

---

## 📋 Prerequisites

### Required APIs

| Service | Purpose | Link |
|---------|---------|------|
| **Zerodha Kite** | Brokerage & data | [kite.trade](https://kite.trade) |
| **Kite Connect API** | Automation | [developers.kite.trade](https://developers.kite.trade) |
| **Telegram Bot** | Alerts | [@BotFather](https://t.me/BotFather) |

### Optional (for enhanced features)

| Service | Feature |
|--------|---------|
| **OpenCode.ai** | AI decision-making |
| **NewsAPI** | Real-time news |
| **Twitter API** | Social sentiment |

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Zerodha Kite
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
KITE_ACCESS_TOKEN=your_token

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Mode
TRADING_MODE=paper  # or "live"

# AI (optional)
ANTHROPIC_AUTH_TOKEN=your_ai_key
ANTHROPIC_BASE_URL=https://opencode.ai/zen
```

### Edit `config.py`

```python
INSTRUMENTS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
MAX_POSITION_SIZE = 100000
STOP_LOSS_PERCENT = 2.0
TARGET_PROFIT_PERCENT = 4.0
SCAN_INTERVAL = 30
```

---

## 📁 Project Structure

```
trading-bot/
├── elite_quant_bot_v5.py     # ⚡ MAIN BOT
├── ai_trader.py            # 🤖 AI Brain
├── config.py              # ⚙️ Config
├── zerodha_client.py      # 📡 Kite API
├── telegram_notifier.py    # 📱 Alerts
├── logger.py              # 📝 Logging
├── setup.py              # 🔑 Token generator
│
├── # 🧠 Advanced Modules
├── rl_genetic_hmm.py        # RL, GA, HMM
├── multi_leg_options.py     # 💵 Options
├── execution_algorithms.py # 📊 Execution
├── nlp_news.py           # 📰 News
├── social_xai_agents.py   # 🌐 Social + XAI
├── risk_pairs_intermarket.py # 📉 Risk
├── emergency_strategies.py # 🛡️ Safety
│
├── .env.example          # 🔐 Template
├── .gitignore          # 🚫 Ignore sensitive files
├── requirements.txt     # 📦 Dependencies
└── README.md         # 📖 Docs
```

---

## 🎯 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    SCAN CYCLE (30s)                        │
├─────────────────────────────────────────────────────────────┤
│  1. Fetch Nifty data                                      │
│  2. Detect regime (HMM + technical)                       │
│  3. Check circuit breakers                               │
│  4. Inter-market analysis                                │
│  5. Multi-agent coordination                             │
│  6. Social/News sentiment                               │
│  7. Generate signals (ensemble)                        │
│  8. Optimize execution (VWAP/TWAP)                      │
│  9. Manage positions                                    │
│  10. Update risk metrics                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Safety First

### Circuit Breakers (Automatic)

| Level | Trigger | Action |
|-------|---------|--------|
| 🟢 LEVEL_1 | -2% | Reduce exposure |
| 🟡 LEVEL_2 | -5% | Close profitable |
| 🔴 LEVEL_3 | -10% | Close all |
| ⚫ LEVEL_4 | Crash | Emergency liquidation |

### Risk Limits

- Max daily trades: **30**
- Max daily loss: **₹10,000**
- Max position: **₹100,000**
- Stop loss: **2%**
- Target: **4%**

---

## 📊 Sample Output

```
══════════════════════════════════════════════════════════════
  SCAN #15 | PnL: ₹+2,500 | Trades: 3 | Accuracy: 67.5%
────────────────────────────────────────────────────────────
  VaR(99%): 2.34% | CVaR: 3.12%
  Multi-Agent: BUY by momentum (75%)
  News: RELIANCE → HOLD | No significant news
  Options: IRON_CONDOR | PoP: 65%
  IV Analysis: IV=18.5% | Rank=MEDIUM
  Calendar: MONDAY_EFFECT | -2.0% adjustment
  ✓ RELIANCE BUY @ ₹2,950 | Target: ₹3,068 | SL: ₹2,891
══════════════════════════════════════════════════════════════
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| No API Key | Run `python setup.py` |
| Market Closed | Bot runs 9:15 AM - 3:30 PM IST |
| Rate Limited | Increase `SCAN_INTERVAL` |
| Import Errors | `pip install -r requirements.txt` |

---

## ⚡ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/Pandas-Forexcellence-150458?style=flat&logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy" />
  <img src="https://img.shields.io/badge/KiteConnect-API-005C47?style=flat" />
  <img src="https://img.shields.io/badge/Telegram-26A5E4?style=flat&logo=telegram" />
</p>

---

## 📜 License

MIT License - See [LICENSE](LICENSE)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" />
</p>

<p align="center">
  <sub>🚀 Trade Smart • Trade Safe • Profit</sub>
</p>

<!--
  _    _   _______          _     _ 
 | |  | | |__   __|        | |   (_)
 | |  | |   | | __ _  ___| |__  _  ___  _ __ 
 | |  | |   | |/ _` |/ __| '_ \| |/ _ \/ '__|
 | |__| |   | | (_| | (__| | | | |  __/ |   
  \____/    |_|\__,_|\___|_| |_|_|\___|_|   
                                        
  _   _  _____  _   _  ____  
 | \ | |/ ____|| | | | |___ \
 |  \| | | (___  | | | |  __) |
 | . ` | \____\| |_| | |__ < 
 |_|\__|_____|\__,_|_|____)_|
 
  _   _  ____         __  __                                                    
 | \ | |/ ___|       |  \/ |                                             
 |  \| | (___  _____| |\/  | ____ _ __ _  ___  ___  ___  _ __ _   _ 
 | . ` | \___ \/ _ \ | | | |/ _` | '__| |/ _ \/ __|/ __|/ _ \ | '__| | | |
 |_|\__||____/|  __/|_| |_|\__,_|_|   |  __/\\__ \\__ \  __/ | |  | |_| |
           |_|   |_|               |_|   |_|   |_|\___/|___/ \___||_|   | .__/|
                                                           | |    
                                                           |_|    
-->