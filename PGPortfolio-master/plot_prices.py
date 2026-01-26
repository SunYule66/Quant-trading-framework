import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# locate the database file robustly
def find_database():
    # 1) try to use package helper if available
    try:
        from pgportfolio.constants import get_database_path
        return get_database_path()
    except Exception:
        pass

    # 2) fallback: look in current directory or immediate subfolders for database/Data.db
    cwd = os.path.abspath(os.path.dirname(__file__))
    candidates = [os.path.join(cwd, 'database', 'Data.db')]
    # also search one level deeper in immediate subdirectories
    for d in os.listdir(cwd):
        sub = os.path.join(cwd, d, 'database', 'Data.db')
        candidates.append(sub)

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError('Data.db not found. Please make sure the database exists under the project database folder (e.g. PGPortfolio-master/PGPortfolio-master/database/Data.db).')


# 连接数据库
db_path = find_database()
conn = sqlite3.connect(db_path)

# 查询所有币种的价格数据
df = pd.read_sql_query('''
    SELECT coin, date, close 
    FROM History 
    ORDER BY coin, date
''', conn)
conn.close()

# 转换时间戳（有些数据可能包含越界时间戳，使用 errors='coerce' 将其转为 NaT，然后丢弃）
df['datetime'] = pd.to_datetime(df['date'], unit='s', errors='coerce')
# 如果全部时间都无法解析，提示用户并退出
if df['datetime'].isna().all():
    raise ValueError('没有可用的时间戳数据：Data.db 中的 date 字段无法解析为正常时间，请检查数据源。')

# 丢弃无法解析的行（越界或无效时间戳）
bad_count = df['datetime'].isna().sum()
if bad_count > 0:
    print(f'警告：丢弃 {bad_count} 条无法解析的时间戳记录（可能越界或格式错误）')
    df = df.dropna(subset=['datetime'])

# 获取所有币种
coins = df['coin'].unique()
print(f'共有 {len(coins)} 个币种: {coins}')

# 创建图表
fig, axes = plt.subplots(len(coins), 1, figsize=(14, 3*len(coins)), sharex=True)
if len(coins) == 1:
    axes = [axes]

for i, coin in enumerate(coins):
    coin_data = df[df['coin'] == coin]
    axes[i].plot(coin_data['datetime'], coin_data['close'], linewidth=0.8)
    axes[i].set_ylabel(f'{coin}')
    axes[i].set_title(f'{coin} Price', fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.xlabel('Date')
plt.tight_layout()
plt.savefig('coin_prices.png', dpi=150)
print('图表已保存到 coin_prices.png')
plt.show()

