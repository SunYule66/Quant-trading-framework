import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARBITRAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ARBITRAGE_DIR, 'data')
RESULTS_DIR = os.path.join(ARBITRAGE_DIR, 'results')

# ==================== 马丁策略 · 数据配置 ====================
# 回测/实时共用，用于数据源、交易对、时间范围等（仪表盘与 --config 可覆盖）
MARTIN_DATA_CONFIG = {
    'exchange': 'okx',              # 交易所：okx / binance
    'symbol': 'BTC-USDT-SWAP',      # 交易对，OKX 用 BTC-USDT-SWAP，Binance 可用 BTCUSDT
    'interval': '1m',               # K 线周期：1m / 5m / 15m / 1h / 4h / 1d
    'start_date': '2026-02-01',     # 回测开始日期
    'end_date': '2026-02-02',       # 回测结束日期
    'binance_futures': False,       # Binance 是否 U 本位合约
    'proxies': {'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'},   # 代理，如 {'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}，None 表示不使用
}

# ==================== 马丁策略 · 策略参数 ====================
MARTIN_STRATEGY_CONFIG = {
    'base_price': None,             # 基准价，None 表示用首个价格
    'grid_spacing_mode': 'atr',     # 网格间距模式：pct / fixed / atr
    'grid_spacing_pct': 0.01,       # 百分比间距（mode=pct 时，如 0.01=1%）
    'grid_spacing_fixed': 100.0,    # 固定点数间距（mode=fixed 时）
    'atr_period': 14,               # ATR 周期（mode=atr 时）
    'atr_multiplier': 1.0,          # ATR 倍数（mode=atr 时）
    'base_size': 0.01,              # 初始手数
    'multiplier': 2.0,              # 马丁倍数
    'max_martin_levels': 8,         # 最大马丁层数
    'normal_levels': 3,              # 普通层数（前 N 层固定手数）
    'max_position_pct': 0.9,        # 最大仓位比例（资金）
    'take_profit_pct': 0.005,       # 止盈比例
    'stop_loss_pct': 0.1,           # 止损比例（总浮亏预警线）
    'take_profit_mode': 'unified',  # 止盈模式：unified / per_trade / layered
    'dynamic_base': True,           # 平仓后是否动态重置基准价
    'total_capital': 10000,         # 总资金（USDT，用于仓位限制）
    'fee_rate': 0.0005,             # 手续费率（如 0.0005=0.05%）
}

# ==================== 马丁策略 · 运行模式 ====================
# 默认运行模式（无 --config 或 JSON 未指定 run_mode 时生效）
# 'backtest' = 回测（历史数据），'realtime' = 实盘/实时（按价格轮询）
RUN_MODE = 'realtime'

# 实时模式轮询间隔（秒）
POLL_INTERVAL_SECONDS = 60

# 实时模式下是否在交易所下单
# True = 模拟盘（使用下方 OKX sandbox / Binance testnet 测试环境下单）
# False = 仅信号不下单；若需真实实盘下单，请将 OKX_CONFIG['sandbox']、BINANCE_CONFIG['testnet'] 设为 False 并自行承担风险
MARTIN_PAPER_TRADE = True

# ==================== 套利策略 · 参数配置 ====================
# 供 method/logic.py 套利系统使用，开平仓阈值与时间窗口
ARBITRAGE_CONFIG = {
    'X': 0.000068,   # 差价触发阈值（百分比）
    'Y': 0.000038,    # 资金费率差触发阈值（百分比）
    'A': 0.000235,    # 可忽略差价阈值
    'B': 0.00014,     # 可忽略资金费率差阈值
    'N': 5,           # 历史小时数
    'M': 5,           # 资金费率不利持续时间（小时）
    'P': 0.0049,      # 盈利平仓阈值
    'Q': 0.000062,    # 亏损止损阈值
}

# ==================== 交易配置（通用） ====================
TRADING_CONFIG = {
    'total_capital': 10000,    # 总资金（USDT）
    'leverage': 10,            # 杠杆倍数
    'check_interval': 60,      # 检查间隔（秒）
    'max_position_ratio': 0.9, # 最大仓位比例（90%）
    'min_balance_ratio': 0.1,  # 最小余额比例（10%，用于保证金）
}

# ==================== 交易所 API 配置 ====================
OKX_CONFIG = {
    'api_key': 'f990715b-34b8-4f79-98bb-ef3bd1993b56',
    'api_secret': 'E5FDAF69E8CDFD753E1B7EE6F45AFFBB',
    'passphrase': 'Sun050301!',
    'sandbox': True,  # True=模拟/测试环境，False=实盘
}

BINANCE_CONFIG = {
    'api_key': 'nKJaUgaS1Vz6CFUGgUU6XZ67pNcA2bFaFhIs4iE7Bufo7dyonh53VN3wfq9BXc5s',
    'api_secret': 'lG6CC63NjdhYOBrz4Dt2gdd1sjkWnPyo2mo8LFCCKGoRNHnqfSodTzCta9mpd7iV',
    'testnet': True,   # True=测试网，False=实盘
}

# ==================== 风险控制配置 ====================
RISK_CONFIG = {
    'max_daily_loss': 0.05,      # 最大日亏损比例（5%）
    'max_single_loss': 0.02,      # 最大单笔亏损比例（2%）
    'max_open_positions': 1,     # 最大同时持仓数
    'emergency_stop_loss': 0.1,  # 紧急止损比例（10%）
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    'level': 'INFO',
    'file': os.path.join(ARBITRAGE_DIR, 'trading.log'),
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}
