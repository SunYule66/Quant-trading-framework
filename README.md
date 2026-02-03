# Quant Trading Framework

本仓库包含 **PGPortfolio** 深度强化学习组合管理框架与 **套利系统**（马丁策略、套利策略、回测与实时交易）。以下为套利系统使用说明。

---

## 套利系统 · 使用介绍

### 1. 目录结构

```
套利系统/
├── config.py              # 统一配置入口（马丁/套利/API/运行模式等）
├── dashboard.py           # Streamlit 可视化管理仪表盘
├── method/
│   ├── martin_bidirectional.py   # 马丁双向网格策略（回测/实时）
│   └── logic.py                   # 套利策略逻辑
├── utils/
│   ├── api_clients.py     # OKX / Binance API 客户端
│   ├── download_data.py  # 数据下载
│   └── ...
├── data/                  # 行情与资金费率数据
├── results/                # 回测结果、图表、运行配置
└── requirements.txt
```

### 2. 配置（config.py）

所有可调参数集中在 `config.py` 中，按模块分区：

| 配置块 | 说明 |
|--------|------|
| **MARTIN_DATA_CONFIG** | 马丁策略数据源：交易所、交易对、K 线周期、日期范围、代理等 |
| **MARTIN_STRATEGY_CONFIG** | 马丁策略参数：网格、手数、止盈止损、马丁倍数等 |
| **RUN_MODE** | 默认运行模式：`backtest` 回测 / `realtime` 实时 |
| **POLL_INTERVAL_SECONDS** | 实时模式轮询间隔（秒） |
| **MARTIN_PAPER_TRADE** | 实时模式是否模拟盘下单（True=测试环境下单） |
| **ARBITRAGE_CONFIG** | 套利策略参数（X/Y/A/B/N/M/P/Q） |
| **OKX_CONFIG / BINANCE_CONFIG** | 交易所 API 与 sandbox/testnet |
| **TRADING_CONFIG / RISK_CONFIG** | 交易与风控参数 |

修改配置后，回测与实时脚本、仪表盘会读取最新配置（仪表盘「配置」页可查看当前值）。

### 3. 安装依赖

```bash
cd 套利系统
pip install -r requirements.txt
```

### 4. 启动仪表盘

```bash
cd 套利系统
streamlit run dashboard.py
```

浏览器访问 `http://localhost:8501`。可在此完成：

- **概览**：KPI、快捷回测、最近结果
- **回测**：配置数据与策略参数，执行马丁双向回测
- **结果**：马丁摘要、交易记录、交互图表（可缩放/悬停）
- **配置**：查看 config.py 中各分区当前值

### 5. 马丁策略 · 命令行

- **回测**（使用 config 或 JSON 配置）：

```bash
cd 套利系统
# 使用 config.py 默认 + RUN_MODE
python method/martin_bidirectional.py

# 使用指定 JSON（如仪表盘生成的 run_config.json）
python method/martin_bidirectional.py --config results/run_config.json
```

JSON 中可指定 `run_mode`（`backtest` / `realtime`）、`data_config`、`strategy_config`、`paper_trade`、`poll_interval_seconds` 等，未写项从 config.py 读取。

- **实时模式**：在 config 中设 `RUN_MODE = 'realtime'`，或在 JSON 中设 `"run_mode": "realtime"`。若无法直连交易所，请在 `MARTIN_DATA_CONFIG['proxies']` 中设置代理（如 `{'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}`）。

### 6. 套利策略

套利逻辑在 `method/logic.py`，参数来自 `config.ARBITRAGE_CONFIG`。在本地运行：

```bash
cd 套利系统
python method/logic.py
```

结果写入 `results/`，可在仪表盘「结果」页选择「套利策略」查看。

### 7. 模拟盘与实盘

- **模拟盘**：`config` 中 `OKX_CONFIG['sandbox']=True`、`BINANCE_CONFIG['testnet']=True`，并设置对应测试网 API Key；马丁实时下单时 `MARTIN_PAPER_TRADE=True` 表示在测试环境下单。
- **实盘**：需将 sandbox/testnet 设为 False、使用实盘 API Key，并自行承担风险。建议先在模拟盘验证。

---

## PGPortfolio（原版）

组合管理与 OKX 数据/实盘部分见子目录 [PGPortfolio-master](PGPortfolio-master/) 及其 [README](PGPortfolio-master/README.md)、[User Guide](PGPortfolio-master/user_guide.md)。

---

## 风险声明

所有策略与实盘/模拟盘交易均存在风险，请自行评估并谨慎使用。
