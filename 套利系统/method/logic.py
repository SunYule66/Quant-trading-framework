import pandas as pd
import csv
import json
import numpy as np
import datetime
from glob import glob
import matplotlib.pyplot as plt
import os
import sys

# 添加父目录到路径，以便导入config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 全项目统一使用北京时间
BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))

class ArbitrageSystem:
    def __init__(self, X, Y, A, B, N, M, P, Q):
        self.X = X #套利差价触发阈值（百分比）
        self.Y = Y #资金费率差触发阈值（百分比）    
        self.A = A #可忽视的差价百分比阈值
        self.B = B #可忽视的资金费率差价百分比阈值
        self.N = N #历史记录数据的小时数量（小时）
        self.M = M #资金费率不利持续时间（小时）
        self.P = P #价差盈利百分比
        self.Q = Q #价差亏损百分比
        self.positions = []

    def direction(self, price_spread, funding_spread):
        # 判断方向是否一致
        return (price_spread >= 0 and funding_spread >= 0) or (price_spread < 0 and funding_spread < 0)

    def check_open(self, df, idx):
        # 获取当前和历史N小时数据
        current = df.iloc[idx]
        current_ts = current.name
        # 以时间窗口（小时）筛选历史
        window_start = current_ts - self.N * 3600
        history = df.loc[(df.index >= window_start) & (df.index < current_ts)]
        # 使用相对价差（百分比）判断 - 永远取正向价差
        price_spread = abs(current['price_a'] - current['price_b']) / min(current['price_a'], current['price_b'])
        funding_spread = current['funding_a'] - current['funding_b']
        # 历史平均价差也使用正向
        if not history.empty:
            history_price_spreads = abs(history['price_a'] - history['price_b']) / pd.concat([history['price_a'], history['price_b']], axis=1).min(axis=1)
            avg_price_spread = history_price_spreads.mean()
        else:
            avg_price_spread = 0
        same_direction = self.direction(current['price_a'] - current['price_b'], funding_spread)

        # 差价套利开仓条件
        if price_spread >= self.X and price_spread > avg_price_spread:
            if same_direction and abs(funding_spread) < self.Y:
                return self.open_position('差价套利', '条件a', current, price_spread, funding_spread, avg_price_spread, '相同')
            elif not same_direction and abs(funding_spread) < self.B:
                return self.open_position('差价套利', '条件b', current, price_spread, funding_spread, avg_price_spread, '不同')

        # 资金费率套利开仓条件 - 包含当前的过去N小时
        recent_N = df.iloc[max(0, idx - self.N + 1):idx + 1]  # 包含当前，共 N 根
        if abs(funding_spread) >= self.Y and (abs(recent_N['funding_a'] - recent_N['funding_b']) >= self.Y).all():
            if same_direction and price_spread < self.X:
                return self.open_position('资金费率套利', '条件a', current, price_spread, funding_spread, avg_price_spread, '相同')
            elif not same_direction and price_spread < self.A:
                return self.open_position('资金费率套利', '条件b', current, price_spread, funding_spread, avg_price_spread, '不同')

        # 组合套利开仓条件
        if same_direction and price_spread >= self.X and price_spread > avg_price_spread and abs(funding_spread) >= self.Y and (abs(recent_N['funding_a'] - recent_N['funding_b']) >= self.Y).all():
            return self.open_position('组合套利', '条件', current, price_spread, funding_spread, avg_price_spread, '相同')
        return None

    def open_position(self, mode, cond, current, price_spread, funding_spread, avg_price_spread, direction):
        # 记录开仓信息并返回，便于后续平仓跟踪
        # price_spread 是正向价差，需要保存原始价差方向用于收益率计算
        original_price_spread = (current['price_a'] - current['price_b']) / current['price_b']
        position = {
            '触发模式': mode,
            '触发条件': cond,
            '开仓差价': price_spread,  # 正向价差
            '开仓原始差价': original_price_spread,  # 原始价差（带符号），用于收益率计算
            '开仓资金费率差': funding_spread,
            '开仓历史平均值': avg_price_spread,
            '开仓方向': direction,
            '开仓价格a': current['price_a'],
            '开仓价格b': current['price_b'],
            '开仓资金费率a': current['funding_a'],
            '开仓资金费率b': current['funding_b'],
            '开仓时间戳': current.name,
            '平仓': False,
            '平仓信息': None
        }
        self.positions.append(position)
        return position

    def check_close(self, df, idx):
        # 遍历所有未平仓的持仓，判断是否满足平仓条件
        closed_any = False
        for pos in self.positions:
            if pos['平仓']:
                continue
            current = df.iloc[idx]
            # 永远取正向价差
            price_spread = abs(current['price_a'] - current['price_b']) / min(current['price_a'], current['price_b'])
            funding_spread = current['funding_a'] - current['funding_b']
            # 差价套利
            if pos['触发模式'] == '差价套利':
                # 条件a（相同方向差价套利）
                if pos['触发条件'] == '条件a':
                    # 平仓条件a: 价格回归盈利
                    if price_spread >= self.P:
                        self.close_position(pos, current, '价格回归盈利')
                        closed_any = True
                        continue
                    # 平仓条件b: 资金费率反转止损
                    # 价差无利可图
                    if price_spread < self.X:
                        # 资金费率方向反转（赚取变成支付）
                        open_side = 1 if pos['开仓资金费率差'] > 0 else -1
                        current_side = 1 if funding_spread > 0 else -1
                        direction_reversed = (open_side != current_side) and (funding_spread != 0)
                        # 资金费率数值 > A 且持续时间 > M
                        if direction_reversed and abs(funding_spread) > self.A:
                            # 检查持续时间
                            close_ts = current.name
                            open_ts = pos['开仓时间戳']
                            hours = (close_ts - open_ts) / 3600
                            if hours >= self.M:
                                self.close_position(pos, current, '资金费率反转止损')
                                closed_any = True
                                continue
                    # 平仓条件c: 价差亏损止损
                    if price_spread >= self.Q:  # 价差亏损是指价差扩大超过阈值
                        self.close_position(pos, current, '价差亏损止损')
                        closed_any = True
                        continue
                # 条件b（不同方向差价套利）
                elif pos['触发条件'] == '条件b':
                    # 平仓条件a: 价格回归盈利
                    if price_spread >= self.P:
                        self.close_position(pos, current, '价格回归盈利')
                        closed_any = True
                        continue
                    # 平仓条件b: 资金费率扩大止损
                    # 价差无利可图
                    if price_spread < self.X:
                        # 需要支付资金费率：原本是 < A，现在变成 > B
                        open_funding_spread = pos['开仓资金费率差']
                        # 检查是否真的变成要支付（资金费率差变为负值或绝对值扩大）
                        # 原本绝对值 < A，现在绝对值 > B，且方向变为需要支付
                        open_abs = abs(open_funding_spread)
                        current_abs = abs(funding_spread)
                        # 原本 < A，现在 > B，且当前需要支付（funding_spread < 0 表示需要支付）
                        if open_abs < self.A and current_abs > self.B and funding_spread < 0:
                            # 检查持续时间
                            close_ts = current.name
                            open_ts = pos['开仓时间戳']
                            hours = (close_ts - open_ts) / 3600
                            if hours >= self.M:
                                self.close_position(pos, current, '资金费率扩大止损')
                                closed_any = True
                                continue
                    # 平仓条件c: 价差亏损止损
                    if price_spread >= self.Q:  # 价差亏损是指价差扩大超过阈值
                        self.close_position(pos, current, '价差亏损止损')
                        closed_any = True
                        continue
            # 资金费率套利
            elif pos['触发模式'] == '资金费率套利':
                open_funding_spread = pos['开仓资金费率差']
                # 平仓条件a: 资金费率收敛或反转
                funding_direction_reversed = (open_funding_spread > 0 and funding_spread < 0) or (open_funding_spread < 0 and funding_spread > 0)
                if abs(funding_spread) < self.B or funding_direction_reversed:
                    self.close_position(pos, current, '资金费率收敛或反转')
                    closed_any = True
                    continue
                # 平仓条件b: 价差盈利平仓
                # 资金费率差仍能赚取（方向未反转）且价差可以实现盈利
                still_earning = (open_funding_spread > 0 and funding_spread > 0) or (open_funding_spread < 0 and funding_spread < 0)
                if still_earning and price_spread >= self.P:
                    self.close_position(pos, current, '价差盈利平仓')
                    closed_any = True
                    continue
                # 平仓条件c: 价差亏损止损
                # 不考虑资金费率，价差无利可图，价差亏损 >= Q
                if price_spread >= self.Q:  # 价差亏损是指价差扩大超过阈值
                    self.close_position(pos, current, '价差亏损止损')
                    closed_any = True
                    continue
            # 组合套利
            elif pos['触发模式'] == '组合套利':
                open_funding_spread = pos['开仓资金费率差']
                funding_direction_reversed = (open_funding_spread > 0 and funding_spread < 0) or (open_funding_spread < 0 and funding_spread > 0)
                # 平仓条件a: 资金费率收敛/反转或价差盈利
                if abs(funding_spread) <= self.B or funding_direction_reversed or price_spread >= self.P:
                    self.close_position(pos, current, '资金费率收敛/反转或价差盈利')
                    closed_any = True
                    continue
                # 平仓条件b: 价差亏损止损
                # 不考虑资金费率，价差无利可图，价差亏损 >= Q
                if price_spread >= self.Q:  # 价差亏损是指价差扩大超过阈值
                    self.close_position(pos, current, '价差亏损止损')
                    closed_any = True
                    continue
        return closed_any

    def has_open_positions(self):
        return any(not p['平仓'] for p in self.positions)

    def close_position(self, pos, current, reason):
        pos['平仓'] = True
        pos['平仓信息'] = {
            '平仓时间戳': current.name,
            '平仓价格a': current['price_a'],
            '平仓价格b': current['price_b'],
            '平仓资金费率a': current['funding_a'],
            '平仓资金费率b': current['funding_b'],
            '平仓原因': reason
        }

    def run(self, df):
        for idx in range(len(df)):
            self.check_open

# 读取OKX价格（BTC-USDT-candlesticks），时间戳除1000
def load_okx_price(filename):
    prices = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row['open_time']) // 1000
                price = float(row['close'])
                prices.append({'timestamp': ts, 'price': price})
            except Exception:
                continue
    return prices

# 读取Binance价格（BTCUSDT-1m），时间戳除1000000
def load_binance_price(filename):
    prices = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                ts = int(row[0]) // 1000000
                price = float(row[4])
                prices.append({'timestamp': ts, 'price': price})
            except Exception:
                continue
    return prices

# 从allswap-fundingrates文件中提取BTC-USDT-SWAP资金费率
def load_okx_funding_from_allswap(filenames):
    """
    从多个allswap-fundingrates文件中提取BTC-USDT-SWAP的资金费率数据
    
    Args:
        filenames: 文件路径列表
        
    Returns:
        list: 资金费率数据列表
    """
    fundings = []
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if row.get('instrument_name') == 'BTC-USDT-SWAP':
                            ts = int(row['funding_time']) // 1000
                            rate = float(row['funding_rate'])
                            fundings.append({'timestamp': ts, 'funding_rate': rate})
                    except Exception:
                        continue
        except Exception:
            continue
    return fundings

# 读取OKX资金费率，时间戳除1000（兼容旧格式）
def load_okx_funding(filename):
    fundings = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row['funding_time']) // 1000
                rate = float(row['funding_rate'])
                fundings.append({'timestamp': ts, 'funding_rate': rate})
            except Exception:
                continue
    return fundings

# 读取Binance资金费率，时间戳为汉字时间
def load_binance_funding(filename):
    fundings = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                contracts = row.get('Contracts', '').strip()
                rate_str = row.get('Funding Rate', '').strip()
                time_str = row.get('Time', '').strip()
                if 'BTCUSDT' in contracts:
                    ts = int(datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp())
                    rate = float(rate_str.replace('%','')) / 100
                    fundings.append({'timestamp': ts, 'funding_rate': rate})
            except Exception:
                continue
    return fundings

# 更新路径指向data文件夹中的子文件夹
glob_okx_price = glob(os.path.join(config.DATA_DIR, "OKX_1m_kline", "BTC-USDT-candlesticks-2025-*.csv"))
glob_binance_price = glob(os.path.join(config.DATA_DIR, "Binance_1m_kline", "BTCUSDT-1m-2025-*.csv"))
glob_okx_funding = glob(os.path.join(config.DATA_DIR, "OKX_funding_rate", "allswap-fundingrates-2025-*.csv"))
file_binance_funding = os.path.join(config.DATA_DIR, "Binance_funding_rate", "Funding Rate History_BTCUSDT Perpetual_2025-12-09.csv")

# 批量读取所有数据文件
okx_prices, binance_prices = [], []
for f in glob_okx_price:
    okx_prices += load_okx_price(f)
for f in glob_binance_price:
    binance_prices += load_binance_price(f)
# 从allswap-fundingrates文件中提取BTC-USDT-SWAP资金费率
okx_fundings = load_okx_funding_from_allswap(glob_okx_funding)
binance_fundings = load_binance_funding(file_binance_funding)

# 构建DataFrame
df_binance_price = pd.DataFrame(binance_prices)
df_okx_price = pd.DataFrame(okx_prices)
df_binance_funding = pd.DataFrame(binance_fundings)
df_okx_funding = pd.DataFrame(okx_fundings)

# 统一用timestamp为索引
df_binance_price.set_index('timestamp', inplace=True)
df_okx_price.set_index('timestamp', inplace=True)
df_binance_funding.set_index('timestamp', inplace=True)
df_okx_funding.set_index('timestamp', inplace=True)

# 合并所有数据，按时间戳对齐
df = pd.DataFrame(index=sorted(set(df_binance_price.index) | set(df_okx_price.index) | set(df_binance_funding.index) | set(df_okx_funding.index)))
df['binance_price'] = df_binance_price['price']
df['okx_price'] = df_okx_price['price']
df['binance_funding'] = df_binance_funding['funding_rate']
df['okx_funding'] = df_okx_funding['funding_rate']

# 时间戳转 datetime（北京时间）
df['datetime'] = pd.to_datetime(df.index, unit='s', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

# 筛选时间段
mask = (df['datetime'] >= pd.Timestamp('2025-11-24')) & (df['datetime'] <= pd.Timestamp('2025-12-07 16:00:00'))
df = df.loc[mask]

# 实例化套利系统，参数统一从 config.ARBITRAGE_CONFIG 读取
system = ArbitrageSystem(**config.ARBITRAGE_CONFIG)

# 预处理数据，构造套利逻辑所需字段
df['price_a'] = df['okx_price']
df['price_b'] = df['binance_price']
df['funding_a'] = df['okx_funding']
df['funding_b'] = df['binance_funding']

# 对齐时间序列并清洗缺失值，避免 NaN 造成开平仓判断异常
df.sort_index(inplace=True)

# 先记录原始可用数据的最后时间戳，再做前向填充，避免前值填充把缺失尾段误判为可用
last_valids = []
for col in ['price_a', 'price_b', 'funding_a', 'funding_b']:
    lv = df[col].last_valid_index()
    if lv is not None:
        last_valids.append(lv)
cutoff = min(last_valids) if last_valids else None

# 再填充，随后按原始可用截止截断
df[['price_a', 'price_b', 'funding_a', 'funding_b']] = df[['price_a', 'price_b', 'funding_a', 'funding_b']].ffill()
if cutoff is not None:
    df = df.loc[:cutoff]
# 截断后再 dropna，避免尾部缺口导致的虚假平仓
df = df.dropna(subset=['price_a', 'price_b', 'funding_a', 'funding_b'])


# 模拟实时交易：持仓未平仓前不再开新仓
for idx in range(len(df)):
    closed_now = system.check_close(df, idx)
    if system.has_open_positions():
        continue
    if closed_now:
        # 本周期刚平仓，为避免同一周期即刻再开仓，跳过本周期
        continue
    system.check_open(df, idx)

# 输出开仓和平仓信息到文件
def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    return obj

# 创建results文件夹
results_dir = config.RESULTS_DIR
os.makedirs(results_dir, exist_ok=True)

# 输出开仓和平仓信息到results文件夹
with open(os.path.join(results_dir, 'arbitrage_positions.json'), 'w', encoding='utf-8') as f:
    json.dump(convert(system.positions), f, ensure_ascii=False, indent=2)

# === 计算总收益率（价格收益率 + 资金费率收益，不含手续费/滑点）===
def calc_price_return(position):
    if not position.get('平仓'):
        return 0.0
    # 使用原始价差（带符号）来判断方向
    open_spread = position.get('开仓原始差价', position['开仓差价'])  # 兼容旧数据
    open_a = position['开仓价格a']
    open_b = position['开仓价格b']
    close_a = position['平仓信息']['平仓价格a']
    close_b = position['平仓信息']['平仓价格b']
    # 假设 spread>0 时开仓为 空a 多b；spread<0 时为 多a 空b
    if open_spread >= 0:
        # 两腿各占50%资金：空a收益率 + 多b收益率
        ret = 0.5 * ((open_a - close_a) / open_a) + 0.5 * ((close_b - open_b) / open_b)
    else:
        # 多a收益率 + 空b收益率
        ret = 0.5 * ((close_a - open_a) / open_a) + 0.5 * ((open_b - close_b) / open_b)
    return float(ret)

def calc_funding_return(position, funding_df):
    if not position.get('平仓'):
        return 0.0
    open_ts = position['开仓时间戳']
    close_ts = position['平仓信息']['平仓时间戳']
    # 提取持仓期间的资金费率序列（含开仓、平仓时刻）
    seg = funding_df.loc[(funding_df.index >= open_ts) & (funding_df.index <= close_ts)]
    if seg.empty:
        return 0.0
    # 使用原始价差（带符号）来判断方向
    open_spread = position.get('开仓原始差价', position['开仓差价'])  # 兼容旧数据
    # spread>0 -> 空a 多b；spread<0 -> 多a 空b
    sign_a = -1 if open_spread >= 0 else 1
    sign_b = -sign_a
    # 资金费率加权：按区间时长（小时）累计；两腿各占50%资金
    seg = seg.sort_index()
    ts_list = list(seg.index) + [close_ts]
    total = 0.0
    for i, ts in enumerate(seg.index):
        next_ts = ts_list[i+1]
        hours = max(0, (next_ts - ts) / 3600)
        rate_eff = 0.5 * seg.loc[ts, 'funding_a'] * sign_a + 0.5 * seg.loc[ts, 'funding_b'] * sign_b
        total += rate_eff * hours
    return float(total)

# 计算并打印总收益率（复利）
funding_df = df[['funding_a', 'funding_b']].copy().ffill().dropna()
cum_return = 1.0
for pos in system.positions:
    price_ret = calc_price_return(pos)
    funding_ret = calc_funding_return(pos, funding_df)
    total_ret = price_ret + funding_ret
    cum_return *= (1 + total_ret)

final_return_pct = (cum_return - 1) * 100
print(f"最终收益率: {final_return_pct:.4f}%")
print(f"结果已保存到: {results_dir}")

# 保存收益率结果到文件
closed_positions = [p for p in system.positions if p.get('平仓', False)]
open_positions = [p for p in system.positions if not p.get('平仓', False)]

results_summary = {
    '最终收益率(%)': final_return_pct,
    '总交易次数': len(closed_positions),
    '未平仓数量': len(open_positions),
    '系统参数': {
        'X': system.X,
        'Y': system.Y,
        'A': system.A,
        'B': system.B,
        'N': system.N,
        'M': system.M,
        'P': system.P,
        'Q': system.Q
    },
    '计算时间': datetime.datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
}
with open(os.path.join(results_dir, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 绘图函数
def plot_results(df, positions, funding_df, output_dir):
    """绘制套利结果图表"""
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据（时间统一为北京时间）
    df_plot = df.copy()
    df_plot['datetime'] = pd.to_datetime(df_plot.index, unit='s', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 子图1: 价格趋势
    plt.subplot(3, 1, 1)
    plt.plot(df_plot['datetime'], df_plot['price_b'], label='Binance 价格')
    plt.plot(df_plot['datetime'], df_plot['price_a'], label='OKX 价格')
    
    # 标注开/平仓点
    try:
        if positions:
            pos_df = pd.DataFrame(positions)
            pos_df['open_ts'] = pd.to_numeric(pos_df['开仓时间戳'], errors='coerce')
            pos_df['close_ts'] = pos_df['平仓信息'].apply(
                lambda x: pd.to_numeric(x.get('平仓时间戳'), errors='coerce') if x and isinstance(x, dict) else None
            )
            pos_df['open_dt'] = pd.to_datetime(pos_df['open_ts'], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            pos_df['close_dt'] = pd.to_datetime(pos_df['close_ts'], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            
            # 准备价格参考数据
            price_ref = df_plot[['datetime', 'price_a']].dropna().copy()
            if not price_ref.empty:
                price_ref['ts'] = (price_ref['datetime'].astype('int64') // 10**9).astype('int64')
                price_ref = price_ref.sort_values('ts')
                ts_min, ts_max = price_ref['ts'].min(), price_ref['ts'].max()
                
                # 开仓点
                open_valid = pos_df.dropna(subset=['open_ts', 'open_dt'])
                open_valid = open_valid[(open_valid['open_ts'] >= ts_min) & (open_valid['open_ts'] <= ts_max)]
                if not open_valid.empty:
                    open_valid = open_valid.copy()
                    open_valid['price'] = open_valid['open_ts'].map(price_ref.set_index('ts')['price_a'])
                    open_valid = open_valid.dropna(subset=['price'])
                    if not open_valid.empty:
                        plt.scatter(open_valid['open_dt'], open_valid['price'], color='green', marker='^', s=60, label='开仓', zorder=5)
                
                # 平仓点
                close_valid = pos_df.dropna(subset=['close_ts', 'close_dt'])
                close_valid = close_valid[(close_valid['close_ts'] >= ts_min) & (close_valid['close_ts'] <= ts_max)]
                if not close_valid.empty:
                    close_valid = close_valid.copy()
                    close_valid['price'] = close_valid['close_ts'].map(price_ref.set_index('ts')['price_a'])
                    close_valid = close_valid.dropna(subset=['price'])
                    if not close_valid.empty:
                        plt.scatter(close_valid['close_dt'], close_valid['price'], color='red', marker='v', s=60, label='平仓', zorder=5)
    except Exception as e:
        print(f"绘图标注错误: {e}")
    
    plt.legend()
    plt.title('BTC-USDT 价格趋势')
    plt.ylabel('价格')
    plt.xlabel('时间')
    
    # 子图2: 资金费率趋势
    plt.subplot(3, 1, 2)
    plt.step(df_plot['datetime'], df_plot['funding_b'], where='post', label='Binance 资金费率')
    plt.step(df_plot['datetime'], df_plot['funding_a'], where='post', label='OKX 资金费率')
    plt.legend()
    plt.title('BTC-USDT 资金费率趋势')
    plt.ylabel('资金费率')
    
    # 子图3: 累计收益率趋势
    plt.subplot(3, 1, 3)
    try:
        pos_df_ret = pd.DataFrame(positions)
        if not pos_df_ret.empty:
            def calc_price_return_plot(row):
                if row.get('平仓信息') is None:
                    return 0.0
                open_a = row['开仓价格a']; open_b = row['开仓价格b']
                close_a = row['平仓信息']['平仓价格a']; close_b = row['平仓信息']['平仓价格b']
                if '开仓原始差价' in row and row['开仓原始差价'] is not None:
                    open_spread = row['开仓原始差价']
                else:
                    open_spread = (open_a - open_b) / open_b
                if open_spread >= 0:
                    return 0.5 * ((open_a - close_a) / open_a) + 0.5 * ((close_b - open_b) / open_b)
                else:
                    return 0.5 * ((close_a - open_a) / open_a) + 0.5 * ((open_b - close_b) / open_b)
            
            def calc_funding_return_plot(row, funding_df_plot):
                if row.get('平仓信息') is None:
                    return 0.0
                open_ts = row['开仓时间戳']; close_ts = row['平仓信息']['平仓时间戳']
                seg = funding_df_plot.loc[(funding_df_plot.index >= open_ts) & (funding_df_plot.index <= close_ts)]
                if seg.empty:
                    return 0.0
                if '开仓原始差价' in row and row['开仓原始差价'] is not None:
                    open_spread = row['开仓原始差价']
                else:
                    open_a = row['开仓价格a']; open_b = row['开仓价格b']
                    open_spread = (open_a - open_b) / open_b
                sign_a = -1 if open_spread >= 0 else 1
                sign_b = -sign_a
                seg = seg.sort_index()
                ts_list = list(seg.index) + [close_ts]
                total = 0.0
                for i, ts in enumerate(seg.index):
                    next_ts = ts_list[i+1]
                    hours = max(0, (next_ts - ts) / 3600)
                    rate_eff = 0.5 * seg.loc[ts, 'funding_a'] * sign_a + 0.5 * seg.loc[ts, 'funding_b'] * sign_b
                    total += rate_eff * hours
                return float(total)
            
            funding_df_plot = funding_df.copy()
            pos_df_ret['close_ts'] = pos_df_ret['平仓信息'].apply(lambda x: x.get('平仓时间戳') if x else None)
            pos_df_ret['close_ts'] = pd.to_numeric(pos_df_ret['close_ts'], errors='coerce')
            ts_min, ts_max = df.index.min(), df.index.max()
            pos_df_ret = pos_df_ret[(pos_df_ret['close_ts'] >= ts_min) & (pos_df_ret['close_ts'] <= ts_max)]
            pos_df_ret['close_dt'] = pd.to_datetime(pos_df_ret['close_ts'], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            pos_df_ret = pos_df_ret.dropna(subset=['close_dt'])
            pos_df_ret['price_ret'] = pos_df_ret.apply(calc_price_return_plot, axis=1)
            pos_df_ret['funding_ret'] = pos_df_ret.apply(lambda r: calc_funding_return_plot(r, funding_df_plot), axis=1)
            pos_df_ret['total_ret'] = pos_df_ret['price_ret'] + pos_df_ret['funding_ret']
            pos_df_ret = pos_df_ret.sort_values('close_dt')
            if not pos_df_ret.empty:
                cum_return_plot = 1.0
                cum_returns = []
                for ret in pos_df_ret['total_ret']:
                    cum_return_plot *= (1 + ret)
                    cum_returns.append((cum_return_plot - 1) * 100)
                pos_df_ret['cum_ret_pct'] = cum_returns
                plt.plot(pos_df_ret['close_dt'], pos_df_ret['cum_ret_pct'], label='累计收益率', linewidth=2)
                plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.title('累计收益率趋势')
                plt.ylabel('收益率 (%)')
    except Exception as e:
        print(f"收益率绘图错误: {e}")
        import traceback
        traceback.print_exc()
    
    plt.xlabel('时间')
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'arbitrage_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()

# 调用绘图函数
plot_results(df, system.positions, funding_df, results_dir)

