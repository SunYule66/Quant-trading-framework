import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import json
import os
import sys

# 添加父目录到路径，以便导入config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 全项目统一使用北京时间
BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

# 截断缺数据的末端：以四列的最后有效时间戳最小值为截止，避免用前值填补无数据的尾段
last_valids = []
for col in ['binance_price', 'okx_price', 'binance_funding', 'okx_funding']:
    lv = df[col].last_valid_index()
    if lv is not None:
        last_valids.append(lv)
cutoff = min(last_valids) if last_valids else None
if cutoff is not None:
    df = df.loc[:cutoff]

# 绘图
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(df['datetime'], df['binance_price'], label='Binance 价格')
plt.plot(df['datetime'], df['okx_price'], label='OKX 价格')
# 标注开/平仓点（使用 OKX 价格作为锚点）
try:
    positions_file = os.path.join(config.RESULTS_DIR, 'arbitrage_positions.json')
    with open(positions_file, 'r', encoding='utf-8') as f:
        positions = json.load(f)
    if positions:
        pos_df = pd.DataFrame(positions)
        # 准备时间戳列
        pos_df['open_ts'] = pd.to_numeric(pos_df['开仓时间戳'], errors='coerce')
        pos_df['close_ts'] = pos_df['平仓信息'].apply(lambda x: pd.to_numeric(x.get('平仓时间戳'), errors='coerce') if x and isinstance(x, dict) else None)
        pos_df['open_dt'] = pd.to_datetime(pos_df['open_ts'], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        pos_df['close_dt'] = pd.to_datetime(pos_df['close_ts'], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

        # 准备价格参考数据
        price_ref = df[['datetime', 'okx_price']].dropna().copy()
        if not price_ref.empty:
            price_ref['ts'] = (price_ref['datetime'].astype('int64') // 10**9).astype('int64')
            price_ref = price_ref.sort_values('ts')
            ts_min, ts_max = price_ref['ts'].min(), price_ref['ts'].max()

            # 过滤开仓点：在时间范围内且有价格数据
            open_valid = pos_df.dropna(subset=['open_ts', 'open_dt'])
            open_valid = open_valid[(open_valid['open_ts'] >= ts_min) & (open_valid['open_ts'] <= ts_max)]
            
            if not open_valid.empty:
                # 直接使用时间戳匹配价格
                open_valid = open_valid.copy()
                open_valid['price'] = open_valid['open_ts'].map(price_ref.set_index('ts')['okx_price'])
                open_valid = open_valid.dropna(subset=['price'])
                if not open_valid.empty:
                    plt.scatter(open_valid['open_dt'], open_valid['price'], color='green', marker='^', s=60, label='开仓', zorder=5)

            # 过滤平仓点：在时间范围内且有价格数据
            close_valid = pos_df.dropna(subset=['close_ts', 'close_dt'])
            close_valid = close_valid[(close_valid['close_ts'] >= ts_min) & (close_valid['close_ts'] <= ts_max)]
            
            if not close_valid.empty:
                close_valid = close_valid.copy()
                close_valid['price'] = close_valid['close_ts'].map(price_ref.set_index('ts')['okx_price'])
                close_valid = close_valid.dropna(subset=['price'])
                if not close_valid.empty:
                    plt.scatter(close_valid['close_dt'], close_valid['price'], color='red', marker='v', s=60, label='平仓', zorder=5)
except Exception as e:
    print(f"绘图标注错误: {e}")
    import traceback
    traceback.print_exc()
plt.legend()
plt.title('BTC-USDT 价格趋势')
plt.ylabel('价格')
plt.xlabel('时间')

plt.subplot(3, 1, 2)
plt.step(df['datetime'], df['binance_funding'], where='post', label='Binance 资金费率')
plt.step(df['datetime'], df['okx_funding'], where='post', label='OKX 资金费率')
plt.legend()
plt.title('BTC-USDT 资金费率趋势')
plt.ylabel('资金费率')

# 收益率变化趋势（按平仓时间累积收益率）
plt.subplot(3, 1, 3)
try:
    positions_file = os.path.join(config.RESULTS_DIR, 'arbitrage_positions.json')
    with open(positions_file, 'r', encoding='utf-8') as f:
        positions_for_ret = json.load(f)
    pos_df_ret = pd.DataFrame(positions_for_ret)
    if not pos_df_ret.empty:
        def calc_price_return(row):
            if row['平仓信息'] is None:
                return 0.0
            open_a = row['开仓价格a']; open_b = row['开仓价格b']
            close_a = row['平仓信息']['平仓价格a']; close_b = row['平仓信息']['平仓价格b']
            # 使用原始价差（带符号）来判断方向，与 logic.py 保持一致
            # 如果存在开仓原始差价，使用它；否则从开仓价格重新计算
            if '开仓原始差价' in row and row['开仓原始差价'] is not None:
                open_spread = row['开仓原始差价']
            else:
                # 从开仓价格重新计算原始价差（带符号）
                open_spread = (open_a - open_b) / open_b
            # 与 logic.py 保持一致：两腿各占50%资金
            if open_spread >= 0:
                return 0.5 * ((open_a - close_a) / open_a) + 0.5 * ((close_b - open_b) / open_b)
            else:
                return 0.5 * ((close_a - open_a) / open_a) + 0.5 * ((open_b - close_b) / open_b)

        def calc_funding_return(row, funding_df):
            if row['平仓信息'] is None:
                return 0.0
            open_ts = row['开仓时间戳']; close_ts = row['平仓信息']['平仓时间戳']
            seg = funding_df.loc[(funding_df.index >= open_ts) & (funding_df.index <= close_ts)]
            if seg.empty:
                return 0.0
            # 使用原始价差（带符号）来判断方向，与 logic.py 保持一致
            # 如果存在开仓原始差价，使用它；否则从开仓价格重新计算
            if '开仓原始差价' in row and row['开仓原始差价'] is not None:
                open_spread = row['开仓原始差价']
            else:
                # 从开仓价格重新计算原始价差（带符号）
                open_a = row['开仓价格a']; open_b = row['开仓价格b']
                open_spread = (open_a - open_b) / open_b
            sign_a = -1 if open_spread >= 0 else 1
            sign_b = -sign_a
            # 与 logic.py 保持一致：按时间加权，两腿各占50%资金
            seg = seg.sort_index()
            ts_list = list(seg.index) + [close_ts]
            total = 0.0
            for i, ts in enumerate(seg.index):
                next_ts = ts_list[i+1]
                hours = max(0, (next_ts - ts) / 3600)
                rate_eff = 0.5 * seg.loc[ts, 'funding_a'] * sign_a + 0.5 * seg.loc[ts, 'funding_b'] * sign_b
                total += rate_eff * hours
            return float(total)

        # 兼容列名（plot_trend 内部使用 okx_funding/binance_funding）
        funding_df = pd.DataFrame({
            'funding_a': df['okx_funding'],
            'funding_b': df['binance_funding'],
        }).ffill().dropna()
        pos_df_ret['close_ts'] = pos_df_ret['平仓信息'].apply(lambda x: x.get('平仓时间戳') if x else None)
        pos_df_ret['close_ts'] = pd.to_numeric(pos_df_ret['close_ts'], errors='coerce')
        # 过滤超出价格数据时间范围的记录，避免在缺数据段计算收益
        ts_min, ts_max = df.index.min(), df.index.max()
        pos_df_ret = pos_df_ret[(pos_df_ret['close_ts'] >= ts_min) & (pos_df_ret['close_ts'] <= ts_max)]
        pos_df_ret['close_dt'] = pd.to_datetime(pos_df_ret['close_ts'], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        pos_df_ret = pos_df_ret.dropna(subset=['close_dt'])
        pos_df_ret['price_ret'] = pos_df_ret.apply(calc_price_return, axis=1)
        pos_df_ret['funding_ret'] = pos_df_ret.apply(lambda r: calc_funding_return(r, funding_df), axis=1)
        pos_df_ret['total_ret'] = pos_df_ret['price_ret'] + pos_df_ret['funding_ret']
        pos_df_ret = pos_df_ret.sort_values('close_dt')
        if not pos_df_ret.empty:
            # 使用复利计算累计收益率（与 logic.py 保持一致）
            cum_return = 1.0
            cum_returns = []
            for ret in pos_df_ret['total_ret']:
                cum_return *= (1 + ret)
                cum_returns.append((cum_return - 1) * 100)
            pos_df_ret['cum_ret_pct'] = cum_returns
            plt.plot(pos_df_ret['close_dt'], pos_df_ret['cum_ret_pct'], label='累计收益率', linewidth=2)
            plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title('累计收益率趋势')
            plt.ylabel('收益率 (%)')
    plt.xlabel('时间')
except Exception:
    pass

plt.tight_layout()
plt.show()