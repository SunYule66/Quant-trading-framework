# ===== 套利系统参数配置 =====
ARBITRAGE_CONFIG = {
    'X': 0.000068,      # 差价触发阈值
    'Y': 0.000038,     # 资金费率差触发阈值
    'A': 0.000235,     # 可忽略差价阈值
    'B': 0.00014,      # 可忽略资金费率差阈值
    'N': 5,            # 历史小时数
    'M': 5,            # 资金费率不利持续时间
    'P': 0.0049,       # 盈利平仓阈值
    'Q': 0.000062,     # 亏损止损阈值
}

# ===== 交易配置 =====
TRADING_CONFIG = {
    'total_capital': 10000,    # 总资金（USDT）
    'leverage': 10,            # 杠杆倍数
    'check_interval': 60,      # 检查间隔（秒）
    'max_position_ratio': 0.9, # 最大仓位比例（90%）
    'min_balance_ratio': 0.1,  # 最小余额比例（10%，用于保证金）
}

# ===== API配置 =====
# OKX API配置
OKX_CONFIG = {
    'api_key': 'f990715b-34b8-4f79-98bb-ef3bd1993b56',
    'api_secret': 'E5FDAF69E8CDFD753E1B7EE6F45AFFBB',
    'passphrase': 'Sun050301!',
    'sandbox': True,  # 测试环境，生产环境设为False
}

# Binance API配置
BINANCE_CONFIG = {
    'api_key': 'nKJaUgaS1Vz6CFUGgUU6XZ67pNcA2bFaFhIs4iE7Bufo7dyonh53VN3wfq9BXc5s',
    'api_secret': 'lG6CC63NjdhYOBrz4Dt2gdd1sjkWnPyo2mo8LFCCKGoRNHnqfSodTzCta9mpd7iV',
    'testnet': True,  # 测试环境，生产环境设为False
}

# ===== 风险控制配置 =====
RISK_CONFIG = {
    'max_daily_loss': 0.05,      # 最大日亏损比例（5%）
    'max_single_loss': 0.02,     # 最大单笔亏损比例（2%）
    'max_open_positions': 1,     # 最大同时持仓数
    'emergency_stop_loss': 0.1,  # 紧急止损比例（10%）
}

# ===== 日志配置 =====
LOG_CONFIG = {
    'level': 'INFO',
    'file': '套利系统/trading.log',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

