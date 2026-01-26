"""
双向马丁网格策略系统

策略流程：
1. 初始化参数（基准价格、网格间距、初始手数、马丁倍数等）
2. 进入主循环（实时价格监控）
3. 当前价格与基准价比较（上涨区间/区间内/下跌区间）
4. 检查网格线触发（上方网格线开买单，下方网格线开卖单）
5. 判断层级（普通层/马丁层）并计算手数
6. 检查止盈条件（统一回本止盈/逐笔小止盈/分层止盈）
7. 检查止损条件（总浮亏达到预警线）
8. 平仓后重置层级和基准价
"""

import pandas as pd
import numpy as np
import json
import os
import datetime
import matplotlib.pyplot as plt
from glob import glob
import csv

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BidirectionalMartinGrid:
    """
    双向马丁网格策略系统
    
    - 基准价格上下设置网格线
    - 价格上涨触发上方网格线 → 开买单（多头）
    - 价格下跌触发下方网格线 → 开卖单（空头）
    - 前N层使用固定手数，之后使用马丁倍数递增
    - 支持三种止盈模式
    """
    
    def __init__(self,
                 base_price=None,              # 基准价格（通常=当前价格，None表示使用第一个价格）
                 grid_spacing_pct=0.01,        # 网格间距（百分比，如0.01表示1%）
                 base_size=0.01,               # 初始手数
                 multiplier=2.0,               # 马丁倍数（通常1.6~3.0）
                 max_martin_levels=8,          # 最大马丁层数（通常4~10）
                 normal_levels=3,              # 普通层数（前N层使用固定手数）
                 max_position_pct=0.9,         # 总最大仓位限制（资金百分比）
                 take_profit_pct=0.005,       # 止盈百分比（小利润）
                 stop_loss_pct=0.1,           # 止损百分比（总浮亏预警线）
                 take_profit_mode='unified',   # 止盈方式：'unified'(统一回本), 'per_trade'(逐笔), 'layered'(分层)
                 dynamic_base=True,            # 是否动态调整基准价（平仓后重置为当前价格）
                 total_capital=10000):         # 总资金（用于仓位限制计算）
        """
        初始化双向马丁网格系统
        
        Args:
            base_price: 基准价格，None表示使用第一个价格
            grid_spacing_pct: 网格间距百分比
            base_size: 初始手数
            multiplier: 马丁倍数
            max_martin_levels: 最大马丁层数
            normal_levels: 普通层数（前N层使用固定手数）
            max_position_pct: 最大仓位限制（资金百分比）
            take_profit_pct: 止盈百分比
            stop_loss_pct: 止损百分比
            take_profit_mode: 止盈模式
            dynamic_base: 是否动态调整基准价
            total_capital: 总资金
        """
        # 参数设置
        self.grid_spacing_pct = grid_spacing_pct
        self.base_size = base_size
        self.multiplier = multiplier
        self.max_martin_levels = max_martin_levels
        self.normal_levels = normal_levels
        self.max_position_pct = max_position_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_mode = take_profit_mode
        self.dynamic_base = dynamic_base
        self.total_capital = total_capital
        
        # 状态变量
        self.base_price = base_price
        self.buy_level = 0          # 买单当前层级
        self.sell_level = 0          # 卖单当前层级
        
        # 持仓记录
        self.buy_positions = []      # 买单持仓列表 [{price, size, level, timestamp}]
        self.sell_positions = []      # 卖单持仓列表
        
        # 交易记录
        self.trade_history = []      # 所有交易记录
        self.total_profit = 0.0      # 累计已实现盈亏
        self.close_history = []      # 平仓历史记录
        
        # 网格线
        self.buy_grid_lines = []     # 上方网格线（买单触发线）
        self.sell_grid_lines = []    # 下方网格线（卖单触发线）
        
        # 状态历史（用于绘图）
        self.state_history = []      # [{timestamp, price, buy_level, sell_level, unrealized_pnl}]
    
    def initialize_base_price(self, current_price):
        """初始化基准价格"""
        if self.base_price is None:
            self.base_price = current_price
            self._update_grid_lines()
            print(f"初始化基准价格: {self.base_price:.2f}")
    
    def _update_grid_lines(self):
        """更新网格线（基准价格上下）"""
        if self.base_price is None:
            return
        
        spacing = self.base_price * self.grid_spacing_pct
        total_levels = self.normal_levels + self.max_martin_levels
        
        # 上方网格线（买单触发线）：base_price + spacing * i
        self.buy_grid_lines = []
        for i in range(1, total_levels + 1):
            line = self.base_price + spacing * i
            self.buy_grid_lines.append(line)
        
        # 下方网格线（卖单触发线）：base_price - spacing * i
        self.sell_grid_lines = []
        for i in range(1, total_levels + 1):
            line = self.base_price - spacing * i
            self.sell_grid_lines.append(line)
    
    def _calculate_position_size(self, level):
        """
        计算指定层级的手数
        
        规则：
        - 普通层（1~normal_levels）：手数 = base_size
        - 马丁层（normal_levels+1起）：手数 = 上层手数 × multiplier
        """
        if level <= self.normal_levels:
            # 普通层：使用固定手数
            return self.base_size
        else:
            # 马丁层：手数 = base_size * multiplier^(level - normal_levels)
            martin_level = level - self.normal_levels
            return self.base_size * (self.multiplier ** martin_level)
    
    def _get_total_position_value(self):
        """计算总仓位价值（用于仓位限制检查）"""
        total = 0.0
        for pos in self.buy_positions:
            total += pos['size'] * pos['price']
        for pos in self.sell_positions:
            total += pos['size'] * pos['price']
        return total
    
    def _check_position_limit(self, new_position_value):
        """检查是否超过最大仓位限制"""
        total_value = self._get_total_position_value()
        if (total_value + new_position_value) / self.total_capital > self.max_position_pct:
            return False
        return True
    
    def _check_buy_grid(self, current_price):
        """
        检查是否触发上方网格线（开买单）
        
        返回: (是否触发, 下一层级, 网格线价格)
        """
        if not self.buy_grid_lines:
            return False, None, None
        
        next_level = self.buy_level + 1
        if next_level > len(self.buy_grid_lines):
            return False, None, None
        
        grid_line = self.buy_grid_lines[next_level - 1]
        
        # 价格上涨到或超过网格线
        if current_price >= grid_line:
            return True, next_level, grid_line
        
        return False, None, None
    
    def _check_sell_grid(self, current_price):
        """
        检查是否触发下方网格线（开卖单）
        
        返回: (是否触发, 下一层级, 网格线价格)
        """
        if not self.sell_grid_lines:
            return False, None, None
        
        next_level = self.sell_level + 1
        if next_level > len(self.sell_grid_lines):
            return False, None, None
        
        grid_line = self.sell_grid_lines[next_level - 1]
        
        # 价格下跌到或低于网格线
        if current_price <= grid_line:
            return True, next_level, grid_line
        
        return False, None, None
    
    def open_buy_position(self, price, timestamp, level):
        """
        开买单（多头）
        
        流程：
        1. 计算手数
        2. 检查仓位限制
        3. 记录持仓
        4. 更新层级
        """
        size = self._calculate_position_size(level)
        new_position_value = size * price
        
        # 检查仓位限制
        if not self._check_position_limit(new_position_value):
            print(f"警告：开买单超过仓位限制，跳过 (level={level}, size={size})")
            return False
        
        position = {
            'price': price,
            'size': size,
            'level': level,
            'timestamp': timestamp,
            'type': 'buy'
        }
        self.buy_positions.append(position)
        self.buy_level = level
        
        # 记录交易
        self.trade_history.append({
            'action': 'open_buy',
            'price': price,
            'size': size,
            'level': level,
            'timestamp': timestamp,
            'grid_line': self.buy_grid_lines[level - 1] if level <= len(self.buy_grid_lines) else None
        })
        
        return True
    
    def open_sell_position(self, price, timestamp, level):
        """
        开卖单（空头）
        
        流程：
        1. 计算手数
        2. 检查仓位限制
        3. 记录持仓
        4. 更新层级
        """
        size = self._calculate_position_size(level)
        new_position_value = size * price
        
        # 检查仓位限制
        if not self._check_position_limit(new_position_value):
            print(f"警告：开卖单超过仓位限制，跳过 (level={level}, size={size})")
            return False
        
        position = {
            'price': price,
            'size': size,
            'level': level,
            'timestamp': timestamp,
            'type': 'sell'
        }
        self.sell_positions.append(position)
        self.sell_level = level
        
        # 记录交易
        self.trade_history.append({
            'action': 'open_sell',
            'price': price,
            'size': size,
            'level': level,
            'timestamp': timestamp,
            'grid_line': self.sell_grid_lines[level - 1] if level <= len(self.sell_grid_lines) else None
        })
        
        return True
    
    def _calculate_unrealized_pnl(self, current_price):
        """
        计算未实现盈亏
        
        买单（多头）：(当前价 - 开仓价) * 数量
        卖单（空头）：(开仓价 - 当前价) * 数量
        """
        pnl = 0.0
        
        for pos in self.buy_positions:
            pnl += (current_price - pos['price']) * pos['size']
        
        for pos in self.sell_positions:
            pnl += (pos['price'] - current_price) * pos['size']
        
        return pnl
    
    def _calculate_average_price(self, positions):
        """计算平均开仓价"""
        if not positions:
            return None
        total_value = sum(p['price'] * p['size'] for p in positions)
        total_size = sum(p['size'] for p in positions)
        return total_value / total_size if total_size > 0 else None
    
    def _check_take_profit(self, current_price):
        """
        检查止盈条件
        
        三种止盈模式：
        1. 统一回本止盈：总净盈亏 ≥ 目标利润
        2. 逐笔小止盈：多空两边都已开仓 → 对冲回本+小利
        3. 分层止盈：不同层级不同止盈比例
        """
        if not self.buy_positions and not self.sell_positions:
            return False, None
        
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        total_invested = sum(p['price'] * p['size'] for p in self.buy_positions + self.sell_positions)
        
        if total_invested == 0:
            return False, None
        
        pnl_pct = unrealized_pnl / total_invested
        
        if self.take_profit_mode == 'unified':
            # 1. 统一回本止盈：总净盈亏 ≥ 目标利润
            if pnl_pct >= self.take_profit_pct:
                return True, 'unified_profit'
        
        elif self.take_profit_mode == 'per_trade':
            # 2. 逐笔小止盈：每组对冲完成后小额获利
            if len(self.buy_positions) > 0 and len(self.sell_positions) > 0:
                buy_avg = self._calculate_average_price(self.buy_positions)
                sell_avg = self._calculate_average_price(self.sell_positions)
                if buy_avg and sell_avg:
                    # 计算对冲盈亏
                    buy_total_size = sum(p['size'] for p in self.buy_positions)
                    sell_total_size = sum(p['size'] for p in self.sell_positions)
                    hedge_size = min(buy_total_size, sell_total_size)
                    hedge_pnl = (sell_avg - buy_avg) * hedge_size
                    hedge_pnl_pct = hedge_pnl / total_invested
                    if hedge_pnl_pct >= self.take_profit_pct:
                        return True, 'hedge_profit'
        
        elif self.take_profit_mode == 'layered':
            # 3. 分层止盈：不同层级不同止盈比例
            # 简化实现：检查是否有单边达到止盈
            if len(self.buy_positions) > 0:
                buy_avg = self._calculate_average_price(self.buy_positions)
                if buy_avg:
                    buy_pnl_pct = (current_price - buy_avg) / buy_avg
                    if buy_pnl_pct >= self.take_profit_pct:
                        return True, 'buy_profit'
            
            if len(self.sell_positions) > 0:
                sell_avg = self._calculate_average_price(self.sell_positions)
                if sell_avg:
                    sell_pnl_pct = (sell_avg - current_price) / sell_avg
                    if sell_pnl_pct >= self.take_profit_pct:
                        return True, 'sell_profit'
        
        return False, None
    
    def _check_stop_loss(self, current_price):
        """
        检查止损条件
        
        总浮亏达到预警线 → 强制全部止损（保护机制）
        """
        if not self.buy_positions and not self.sell_positions:
            return False
        
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        total_invested = sum(p['price'] * p['size'] for p in self.buy_positions + self.sell_positions)
        
        if total_invested == 0:
            return False
        
        pnl_pct = unrealized_pnl / total_invested
        
        # 总浮亏达到预警线
        if pnl_pct <= -self.stop_loss_pct:
            return True
        
        return False
    
    def close_all_positions(self, current_price, timestamp, reason):
        """
        平仓所有持仓
        
        平仓后：
        - 重置该方向层级 → 0
        - 可选择：基准价跟随当前价格重新设定（动态基准）或保持原基准（固定区间）
        """
        if not self.buy_positions and not self.sell_positions:
            return None
        
        # 计算已实现盈亏
        realized_pnl = self._calculate_unrealized_pnl(current_price)
        self.total_profit += realized_pnl
        
        # 记录平仓信息
        close_info = {
            'timestamp': timestamp,
            'price': current_price,
            'reason': reason,
            'buy_positions_count': len(self.buy_positions),
            'sell_positions_count': len(self.sell_positions),
            'buy_level': self.buy_level,
            'sell_level': self.sell_level,
            'realized_pnl': realized_pnl,
            'total_profit': self.total_profit,
            'buy_avg_price': self._calculate_average_price(self.buy_positions),
            'sell_avg_price': self._calculate_average_price(self.sell_positions)
        }
        
        self.close_history.append(close_info)
        
        # 记录交易历史
        self.trade_history.append({
            'action': 'close_all',
            'price': current_price,
            'timestamp': timestamp,
            'reason': reason,
            'realized_pnl': realized_pnl,
            'buy_count': len(self.buy_positions),
            'sell_count': len(self.sell_positions)
        })
        
        # 清空持仓
        self.buy_positions = []
        self.sell_positions = []
        
        # 重置层级
        self.buy_level = 0
        self.sell_level = 0
        
        # 如果启用动态基准，更新基准价
        if self.dynamic_base:
            old_base = self.base_price
            self.base_price = current_price
            self._update_grid_lines()
            print(f"平仓后更新基准价: {old_base:.2f} → {self.base_price:.2f} (原因: {reason})")
        
        return close_info
    
    def process_price(self, current_price, timestamp):
        """
        处理当前价格（主循环逻辑）
        
        流程：
        1. 初始化基准价格（如果未初始化）
        2. 检查止损
        3. 检查止盈
        4. 检查网格线触发（上方/下方）
        5. 记录状态历史
        """
        # 初始化基准价格
        if self.base_price is None:
            self.initialize_base_price(current_price)
        
        # 计算未实现盈亏
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        
        # 记录状态历史
        self.state_history.append({
            'timestamp': timestamp,
            'price': current_price,
            'buy_level': self.buy_level,
            'sell_level': self.sell_level,
            'buy_count': len(self.buy_positions),
            'sell_count': len(self.sell_positions),
            'unrealized_pnl': unrealized_pnl,
            'total_profit': self.total_profit
        })
        
        # 检查止损（优先）
        if self._check_stop_loss(current_price):
            self.close_all_positions(current_price, timestamp, 'stop_loss')
            return
        
        # 检查止盈
        should_close, close_reason = self._check_take_profit(current_price)
        if should_close:
            self.close_all_positions(current_price, timestamp, close_reason)
            return
        
        # 检查网格线触发
        # 检查上方网格线（买单）
        buy_triggered, buy_level, buy_grid_line = self._check_buy_grid(current_price)
        if buy_triggered:
            self.open_buy_position(current_price, timestamp, buy_level)
        
        # 检查下方网格线（卖单）
        sell_triggered, sell_level, sell_grid_line = self._check_sell_grid(current_price)
        if sell_triggered:
            self.open_sell_position(current_price, timestamp, sell_level)
    
    def get_status(self, current_price=None):
        """获取当前状态"""
        status = {
            'base_price': self.base_price,
            'buy_level': self.buy_level,
            'sell_level': self.sell_level,
            'buy_positions_count': len(self.buy_positions),
            'sell_positions_count': len(self.sell_positions),
            'total_profit': self.total_profit,
            'trade_count': len(self.trade_history)
        }
        
        if current_price is not None:
            status['current_price'] = current_price
            status['unrealized_pnl'] = self._calculate_unrealized_pnl(current_price)
            if self.buy_positions:
                status['buy_avg_price'] = self._calculate_average_price(self.buy_positions)
            if self.sell_positions:
                status['sell_avg_price'] = self._calculate_average_price(self.sell_positions)
        
        return status


# ========== 数据加载函数 ==========

def load_price_data(filename):
    """加载价格数据（支持多种格式）"""
    prices = []
    file_loaded = False
    
    # 首先尝试Binance格式（无表头的CSV）
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # 读取第一行判断格式
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            # 判断是否为Binance格式（纯数字，用逗号分隔，第一列是时间戳）
            if ',' in first_line and first_line.split(',')[0].isdigit():
                reader = csv.reader(f)
                for row_num, row in enumerate(reader, 1):
                    try:
                        if len(row) < 5:
                            continue
                        # Binance格式：时间戳(微秒)在第0列，收盘价在第4列
                        ts_microseconds = int(row[0])
                        ts = ts_microseconds // 1000000  # 转换为秒
                        price = float(row[4])
                        if ts > 0 and price > 0:
                            prices.append({'timestamp': ts, 'price': price})
                    except (ValueError, IndexError) as e:
                        # 跳过无效行，但不中断加载
                        continue
                file_loaded = True
    except Exception as e:
        pass
    
    # 如果Binance格式失败，尝试OKX格式（有表头的CSV）
    if not file_loaded:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # OKX格式：open_time字段，时间戳需要除以1000
                        ts = int(row.get('open_time', row.get('timestamp', 0))) // 1000
                        price = float(row.get('close', row.get('price', 0)))
                        if ts > 0 and price > 0:
                            prices.append({'timestamp': ts, 'price': price})
                    except (ValueError, KeyError):
                        continue
                file_loaded = True
        except Exception as e:
            pass
    
    return prices


# ========== 主程序 ==========

if __name__ == '__main__':
    # ===== 参数配置 =====
    config = {
        'base_price': None,              # None表示使用第一个价格作为基准
        'grid_spacing_pct': 0.01,        # 1%网格间距
        'base_size': 0.01,               # 初始手数
        'multiplier': 2.0,               # 马丁倍数
        'max_martin_levels': 8,          # 最大马丁层数
        'normal_levels': 3,              # 普通层数（前3层使用固定手数）
        'max_position_pct': 0.9,         # 最大仓位限制（90%）
        'take_profit_pct': 0.005,        # 0.5%止盈
        'stop_loss_pct': 0.1,            # 10%止损
        'take_profit_mode': 'unified',    # 统一回本止盈
        'dynamic_base': True,             # 动态基准价
        'total_capital': 10000           # 总资金
    }
    
    print("=" * 60)
    print("双向马丁网格策略系统")
    print("=" * 60)
    print(f"参数配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # ===== 加载数据 =====
    print("\n加载价格数据...")
    price_files = glob("套利系统/data/Binance_1m_kline/BTCUSDT-1m-2025-*.csv")
    if not price_files:
        print("未找到价格数据文件")
        print(f"搜索路径: 套利系统/data/Binance_1m_kline/BTCUSDT-1m-2025-*.csv")
        exit(1)
    
    print(f"找到 {len(price_files)} 个数据文件")
    all_prices = []
    for i, f in enumerate(price_files, 1):
        try:
            file_prices = load_price_data(f)
            all_prices += file_prices
            print(f"  [{i}/{len(price_files)}] {os.path.basename(f)}: 加载了 {len(file_prices)} 条数据")
        except Exception as e:
            print(f"  [{i}/{len(price_files)}] {os.path.basename(f)}: 加载失败 - {e}")
            continue
    
    if not all_prices:
        print("错误：未能加载任何价格数据")
        print("请检查数据文件格式和路径")
        exit(1)
    
    print(f"总计加载了 {len(all_prices)} 条价格数据")
    
    # 构建DataFrame
    df = pd.DataFrame(all_prices)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df.drop_duplicates()
    
    # 筛选时间段
    df['datetime'] = pd.to_datetime(df.index, unit='s')
    mask = (df['datetime'] >= pd.Timestamp('2025-11-24')) & (df['datetime'] <= pd.Timestamp('2025-12-07 16:00:00'))
    df = df.loc[mask]
    df = df.dropna(subset=['price'])
    
    if df.empty:
        print("筛选后的数据为空")
        exit(1)
    
    print(f"加载了 {len(df)} 条价格数据")
    print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    print(f"价格范围: {df['price'].min():.2f} 至 {df['price'].max():.2f}")
    
    # ===== 初始化系统 =====
    print("\n初始化双向马丁网格系统...")
    system = BidirectionalMartinGrid(**config)
    
    # ===== 回测 =====
    print("\n开始回测...")
    for idx, row in df.iterrows():
        system.process_price(row['price'], idx)
    
    # 最终平仓（如果有持仓）
    if system.buy_positions or system.sell_positions:
        final_price = df.iloc[-1]['price']
        final_timestamp = df.index[-1]
        system.close_all_positions(final_price, final_timestamp, 'end_of_data')
    
    # ===== 保存结果 =====
    results_dir = '套利系统/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存交易历史
    with open(os.path.join(results_dir, 'martin_bidirectional_trades.json'), 'w', encoding='utf-8') as f:
        json.dump(system.trade_history, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存平仓历史
    with open(os.path.join(results_dir, 'martin_bidirectional_closes.json'), 'w', encoding='utf-8') as f:
        json.dump(system.close_history, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存结果摘要
    buy_trades = len([t for t in system.trade_history if t['action'] == 'open_buy'])
    sell_trades = len([t for t in system.trade_history if t['action'] == 'open_sell'])
    close_trades = len([t for t in system.trade_history if t['action'] == 'close_all'])
    
    results_summary = {
        '策略类型': '双向马丁网格',
        '最终盈亏': system.total_profit,
        '总交易次数': len(system.trade_history),
        '开买单次数': buy_trades,
        '开卖单次数': sell_trades,
        '平仓次数': close_trades,
        '系统参数': config,
        '回测时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '数据时间范围': {
            '开始': str(df['datetime'].min()),
            '结束': str(df['datetime'].max())
        }
    }
    
    with open(os.path.join(results_dir, 'martin_bidirectional_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60)
    print(f"最终盈亏: {system.total_profit:.4f} USDT")
    print(f"总交易次数: {len(system.trade_history)}")
    print(f"开买单次数: {buy_trades}")
    print(f"开卖单次数: {sell_trades}")
    print(f"平仓次数: {close_trades}")
    print(f"结果已保存到: {results_dir}")
    print("=" * 60)
    
    # ===== 绘图 =====
    def plot_martin_bidirectional_results(df, system, output_dir):
        """绘制双向马丁网格策略结果"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # 子图1: 价格趋势和交易点
        ax1 = axes[0]
        ax1.plot(df['datetime'], df['price'], label='价格', linewidth=1.5, alpha=0.7, color='black')
        
        # 绘制基准价
        if system.base_price:
            ax1.axhline(y=system.base_price, color='orange', linestyle='--', linewidth=1, label='基准价', alpha=0.7)
        
        # 标注交易点
        buy_trades = [t for t in system.trade_history if t['action'] == 'open_buy']
        sell_trades = [t for t in system.trade_history if t['action'] == 'open_sell']
        close_trades = [t for t in system.trade_history if t['action'] == 'close_all']
        
        if buy_trades:
            buy_times = [pd.to_datetime(t['timestamp'], unit='s') for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=80, label='开买单', zorder=5, alpha=0.8)
        
        if sell_trades:
            sell_times = [pd.to_datetime(t['timestamp'], unit='s') for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=80, label='开卖单', zorder=5, alpha=0.8)
        
        if close_trades:
            close_times = [pd.to_datetime(t['timestamp'], unit='s') for t in close_trades]
            close_prices = [t['price'] for t in close_trades]
            ax1.scatter(close_times, close_prices, color='blue', marker='x', s=150, label='平仓', zorder=5, alpha=0.9, linewidths=2)
        
        ax1.set_title('价格趋势与交易点', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 累计盈亏
        ax2 = axes[1]
        if system.state_history:
            state_df = pd.DataFrame(system.state_history)
            state_df['datetime'] = pd.to_datetime(state_df['timestamp'], unit='s')
            state_df = state_df.sort_values('datetime')
            
            # 计算累计盈亏（已实现 + 未实现）
            state_df['total_pnl'] = state_df['total_profit'] + state_df['unrealized_pnl']
            
            ax2.plot(state_df['datetime'], state_df['total_pnl'], label='累计盈亏', linewidth=2, color='blue')
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.fill_between(state_df['datetime'], 0, state_df['total_pnl'], 
                            where=(state_df['total_pnl'] >= 0), alpha=0.3, color='green', label='盈利区间')
            ax2.fill_between(state_df['datetime'], 0, state_df['total_pnl'], 
                            where=(state_df['total_pnl'] < 0), alpha=0.3, color='red', label='亏损区间')
        
        ax2.set_title('累计盈亏趋势', fontsize=14, fontweight='bold')
        ax2.set_ylabel('累计盈亏 (USDT)', fontsize=11)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 持仓层级变化
        ax3 = axes[2]
        if system.state_history:
            ax3.plot(state_df['datetime'], state_df['buy_level'], label='买单层级', linewidth=2, color='green', alpha=0.8)
            ax3.plot(state_df['datetime'], state_df['sell_level'], label='卖单层级', linewidth=2, color='red', alpha=0.8)
            ax3.fill_between(state_df['datetime'], 0, state_df['buy_level'], alpha=0.2, color='green')
            ax3.fill_between(state_df['datetime'], 0, state_df['sell_level'], alpha=0.2, color='red')
        
        ax3.set_title('持仓层级变化', fontsize=14, fontweight='bold')
        ax3.set_ylabel('层级', fontsize=11)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 持仓数量变化
        ax4 = axes[3]
        if system.state_history:
            ax4.plot(state_df['datetime'], state_df['buy_count'], label='买单持仓数', linewidth=2, color='green', alpha=0.8)
            ax4.plot(state_df['datetime'], state_df['sell_count'], label='卖单持仓数', linewidth=2, color='red', alpha=0.8)
            ax4.fill_between(state_df['datetime'], 0, state_df['buy_count'], alpha=0.2, color='green')
            ax4.fill_between(state_df['datetime'], 0, state_df['sell_count'], alpha=0.2, color='red')
        
        ax4.set_title('持仓数量变化', fontsize=14, fontweight='bold')
        ax4.set_xlabel('时间', fontsize=11)
        ax4.set_ylabel('持仓数量', fontsize=11)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'martin_bidirectional_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        plt.close()
    
    # 调用绘图函数
    try:
        print("\n生成图表...")
        plot_martin_bidirectional_results(df, system, results_dir)
    except Exception as e:
        print(f"绘图时出错: {e}")
        import traceback
        traceback.print_exc()

