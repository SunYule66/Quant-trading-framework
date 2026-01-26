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


class MartinGridSystem:
    """
    双向马丁网格策略系统
    
    策略逻辑：
    1. 在基准价格上下设置网格线
    2. 价格上涨到上方网格线时开买单（多头）
    3. 价格下跌到下方网格线时开卖单（空头）
    4. 前N层使用固定手数，之后使用马丁倍数递增
    5. 满足止盈条件时平仓
    """
    
    def __init__(self, 
                 base_price=None,
                 grid_spacing_pct=0.01,  # 网格间距（百分比）
                 base_size=0.01,  # 初始手数
                 multiplier=2.0,  # 马丁倍数
                 max_martin_levels=5,  # 最大马丁层数
                 normal_levels=3,  # 普通层数（前N层使用固定手数）
                 max_position_pct=0.9,  # 最大仓位限制（资金百分比）
                 take_profit_pct=0.005,  # 止盈百分比
                 stop_loss_pct=0.1,  # 止损百分比（总浮亏）
                 take_profit_mode='unified',  # 止盈模式：'unified'(统一回本), 'per_trade'(逐笔), 'layered'(分层)
                 dynamic_base=True):  # 是否动态调整基准价
        """
        初始化双向马丁网格系统
        
        Args:
            base_price: 基准价格（如果为None，使用第一个价格）
            grid_spacing_pct: 网格间距百分比（如0.01表示1%）
            base_size: 初始手数
            multiplier: 马丁倍数（每层手数 = 上层手数 × multiplier）
            max_martin_levels: 最大马丁层数
            normal_levels: 普通层数（前N层使用固定手数base_size）
            max_position_pct: 最大仓位限制（资金百分比）
            take_profit_pct: 止盈百分比
            stop_loss_pct: 止损百分比（总浮亏达到此值时强制止损）
            take_profit_mode: 止盈模式
            dynamic_base: 是否动态调整基准价（平仓后重置为当前价格）
        """
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
        
        # 初始化状态
        self.base_price = base_price
        self.buy_level = 0  # 买单当前层级
        self.sell_level = 0  # 卖单当前层级
        self.buy_positions = []  # 买单持仓列表 [{price, size, level, timestamp}]
        self.sell_positions = []  # 卖单持仓列表
        self.trade_history = []  # 交易历史
        self.total_profit = 0.0  # 累计盈亏
        self.level_history = []  # 层级历史记录 [{timestamp, buy_level, sell_level, price}]
        
        # 网格线缓存
        self.buy_grid_lines = []  # 上方网格线（买单触发线）
        self.sell_grid_lines = []  # 下方网格线（卖单触发线）
        
    def initialize_base_price(self, current_price):
        """初始化基准价格"""
        if self.base_price is None:
            self.base_price = current_price
            self._update_grid_lines()
    
    def _update_grid_lines(self):
        """更新网格线"""
        if self.base_price is None:
            return
        
        spacing = self.base_price * self.grid_spacing_pct
        
        # 上方网格线（买单触发线）
        self.buy_grid_lines = []
        for i in range(1, self.max_martin_levels + self.normal_levels + 1):
            line = self.base_price + spacing * i
            self.buy_grid_lines.append(line)
        
        # 下方网格线（卖单触发线）
        self.sell_grid_lines = []
        for i in range(1, self.max_martin_levels + self.normal_levels + 1):
            line = self.base_price - spacing * i
            self.sell_grid_lines.append(line)
    
    def _calculate_position_size(self, level):
        """计算指定层级的手数"""
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
    
    def _check_buy_grid(self, current_price):
        """检查是否触发上方网格线（开买单）"""
        if not self.buy_grid_lines:
            return False
        
        # 检查是否达到下一层网格线
        next_level = self.buy_level + 1
        if next_level > len(self.buy_grid_lines):
            return False
        
        grid_line = self.buy_grid_lines[next_level - 1]
        
        # 价格上涨到或超过网格线
        if current_price >= grid_line:
            return True, next_level, grid_line
        return False, None, None
    
    def _check_sell_grid(self, current_price):
        """检查是否触发下方网格线（开卖单）"""
        if not self.sell_grid_lines:
            return False
        
        # 检查是否达到下一层网格线
        next_level = self.sell_level + 1
        if next_level > len(self.sell_grid_lines):
            return False
        
        grid_line = self.sell_grid_lines[next_level - 1]
        
        # 价格下跌到或低于网格线
        if current_price <= grid_line:
            return True, next_level, grid_line
        return False, None, None
    
    def open_buy_position(self, price, timestamp, level):
        """开买单（多头）"""
        size = self._calculate_position_size(level)
        
        # 检查仓位限制
        total_value = self._get_total_position_value()
        new_position_value = size * price
        # 这里简化处理，实际应该基于总资金计算
        # if (total_value + new_position_value) / total_capital > self.max_position_pct:
        #     return False
        
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
            'timestamp': timestamp
        })
        
        return True
    
    def open_sell_position(self, price, timestamp, level):
        """开卖单（空头）"""
        size = self._calculate_position_size(level)
        
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
            'timestamp': timestamp
        })
        
        return True
    
    def _calculate_unrealized_pnl(self, current_price):
        """计算未实现盈亏"""
        pnl = 0.0
        
        # 买单盈亏（多头）：(当前价 - 开仓价) * 数量
        for pos in self.buy_positions:
            pnl += (current_price - pos['price']) * pos['size']
        
        # 卖单盈亏（空头）：(开仓价 - 当前价) * 数量
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
        """检查止盈条件"""
        if not self.buy_positions and not self.sell_positions:
            return False, None
        
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        total_invested = sum(p['price'] * p['size'] for p in self.buy_positions + self.sell_positions)
        
        if total_invested == 0:
            return False, None
        
        pnl_pct = unrealized_pnl / total_invested
        
        if self.take_profit_mode == 'unified':
            # 统一回本止盈：总盈亏 >= 目标利润
            if pnl_pct >= self.take_profit_pct:
                return True, 'unified_profit'
        
        elif self.take_profit_mode == 'per_trade':
            # 逐笔小止盈：每组对冲完成后小额获利
            if len(self.buy_positions) > 0 and len(self.sell_positions) > 0:
                # 多空对冲，计算对冲盈亏
                buy_avg = self._calculate_average_price(self.buy_positions)
                sell_avg = self._calculate_average_price(self.sell_positions)
                if buy_avg and sell_avg:
                    hedge_pnl = (sell_avg - buy_avg) * min(
                        sum(p['size'] for p in self.buy_positions),
                        sum(p['size'] for p in self.sell_positions)
                    )
                    hedge_pnl_pct = hedge_pnl / total_invested
                    if hedge_pnl_pct >= self.take_profit_pct:
                        return True, 'hedge_profit'
        
        elif self.take_profit_mode == 'layered':
            # 分层止盈：不同层级不同止盈比例
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
        """检查止损条件"""
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
        """平仓所有持仓"""
        if not self.buy_positions and not self.sell_positions:
            return
        
        # 计算盈亏
        realized_pnl = self._calculate_unrealized_pnl(current_price)
        self.total_profit += realized_pnl
        
        # 记录平仓信息
        close_info = {
            'timestamp': timestamp,
            'price': current_price,
            'reason': reason,
            'buy_positions_count': len(self.buy_positions),
            'sell_positions_count': len(self.sell_positions),
            'realized_pnl': realized_pnl,
            'total_profit': self.total_profit
        }
        
        # 记录交易历史
        self.trade_history.append({
            'action': 'close_all',
            'price': current_price,
            'timestamp': timestamp,
            'reason': reason,
            'realized_pnl': realized_pnl
        })
        
        # 清空持仓
        self.buy_positions = []
        self.sell_positions = []
        
        # 重置层级
        self.buy_level = 0
        self.sell_level = 0
        
        # 如果启用动态基准，更新基准价
        if self.dynamic_base:
            self.base_price = current_price
            self._update_grid_lines()
        
        return close_info
    
    def process_price(self, current_price, timestamp):
        """处理当前价格（主循环逻辑）"""
        # 初始化基准价格
        if self.base_price is None:
            self.initialize_base_price(current_price)
        
        # 记录层级历史
        self.level_history.append({
            'timestamp': timestamp,
            'price': current_price,
            'buy_level': self.buy_level,
            'sell_level': self.sell_level
        })
        
        # 检查止损
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


# ========== 数据加载和回测 ==========

def load_price_data(filename):
    """加载价格数据"""
    prices = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(row.get('open_time', row.get('timestamp', 0))) // 1000
                    price = float(row.get('close', row.get('price', 0)))
                    prices.append({'timestamp': ts, 'price': price})
                except:
                    continue
    except:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        ts = int(row[0]) // 1000000
                        price = float(row[4])
                        prices.append({'timestamp': ts, 'price': price})
                    except:
                        continue
        except:
            pass
    return prices


# ========== 主程序 ==========

if __name__ == '__main__':
    # 参数配置
    config = {
        'base_price': None,  # None表示使用第一个价格作为基准
        'grid_spacing_pct': 0.01,  # 1%网格间距
        'base_size': 0.01,  # 初始手数
        'multiplier': 2.0,  # 马丁倍数
        'max_martin_levels': 5,  # 最大马丁层数
        'normal_levels': 3,  # 普通层数
        'max_position_pct': 0.9,  # 最大仓位限制
        'take_profit_pct': 0.005,  # 0.5%止盈
        'stop_loss_pct': 0.1,  # 10%止损
        'take_profit_mode': 'unified',  # 统一回本止盈
        'dynamic_base': True  # 动态基准价
    }
    
    # 加载数据（使用Binance数据作为示例）
    price_files = glob("套利系统/data/Binance_1m_kline/BTCUSDT-1m-2025-*.csv")
    all_prices = []
    for f in price_files:
        all_prices += load_price_data(f)
    
    if not all_prices:
        print("未找到价格数据文件")
        exit(1)
    
    # 构建DataFrame
    df = pd.DataFrame(all_prices)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # 筛选时间段
    df['datetime'] = pd.to_datetime(df.index, unit='s')
    mask = (df['datetime'] >= pd.Timestamp('2025-11-24')) & (df['datetime'] <= pd.Timestamp('2025-12-07 16:00:00'))
    df = df.loc[mask]
    df = df.dropna(subset=['price'])
    
    if df.empty:
        print("筛选后的数据为空")
        exit(1)
    
    # 初始化马丁网格系统
    system = MartinGridSystem(**config)
    
    # 回测
    print("开始回测...")
    for idx, row in df.iterrows():
        system.process_price(row['price'], idx)
    
    # 最终平仓（如果有持仓）
    if system.buy_positions or system.sell_positions:
        final_price = df.iloc[-1]['price']
        final_timestamp = df.index[-1]
        system.close_all_positions(final_price, final_timestamp, 'end_of_data')
    
    # 保存结果
    results_dir = '套利系统/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存交易历史
    with open(os.path.join(results_dir, 'martin_grid_trades.json'), 'w', encoding='utf-8') as f:
        json.dump(system.trade_history, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存结果摘要
    results_summary = {
        '策略类型': '双向马丁网格',
        '最终盈亏': system.total_profit,
        '总交易次数': len(system.trade_history),
        '系统参数': config,
        '回测时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(results_dir, 'martin_grid_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print(f"回测完成！")
    print(f"最终盈亏: {system.total_profit:.4f}")
    print(f"总交易次数: {len(system.trade_history)}")
    print(f"结果已保存到: {results_dir}")
    
    # 绘图
    def plot_martin_grid_results(df, system, output_dir):
        """绘制马丁网格策略结果"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 子图1: 价格趋势和网格线
        ax1 = axes[0]
        ax1.plot(df['datetime'], df['price'], label='价格', linewidth=1.5, alpha=0.7)
        
        # 绘制基准价
        if system.base_price:
            ax1.axhline(y=system.base_price, color='orange', linestyle='--', linewidth=1, label='基准价', alpha=0.7)
        
        # 标注开仓点
        buy_trades = [t for t in system.trade_history if t['action'] == 'open_buy']
        sell_trades = [t for t in system.trade_history if t['action'] == 'open_sell']
        close_trades = [t for t in system.trade_history if t['action'] == 'close_all']
        
        if buy_trades:
            buy_times = [pd.to_datetime(t['timestamp'], unit='s') for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=50, label='开买单', zorder=5, alpha=0.7)
        
        if sell_trades:
            sell_times = [pd.to_datetime(t['timestamp'], unit='s') for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=50, label='开卖单', zorder=5, alpha=0.7)
        
        if close_trades:
            close_times = [pd.to_datetime(t['timestamp'], unit='s') for t in close_trades]
            close_prices = [t['price'] for t in close_trades]
            ax1.scatter(close_times, close_prices, color='blue', marker='x', s=100, label='平仓', zorder=5, alpha=0.8)
        
        ax1.set_title('价格趋势与交易点', fontsize=12, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=10)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 累计盈亏
        ax2 = axes[1]
        cumulative_pnl = []
        cumulative_profit = 0.0
        
        # 按时间顺序计算累计盈亏
        trade_df = pd.DataFrame(system.trade_history)
        if not trade_df.empty:
            trade_df['datetime'] = pd.to_datetime(trade_df['timestamp'], unit='s')
            trade_df = trade_df.sort_values('datetime')
            
            for _, trade in trade_df.iterrows():
                if trade['action'] == 'close_all':
                    cumulative_profit += trade.get('realized_pnl', 0)
                cumulative_pnl.append({
                    'datetime': trade['datetime'],
                    'cumulative_pnl': cumulative_profit
                })
        
        if cumulative_pnl:
            pnl_df = pd.DataFrame(cumulative_pnl)
            ax2.plot(pnl_df['datetime'], pnl_df['cumulative_pnl'], label='累计盈亏', linewidth=2, color='blue')
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.fill_between(pnl_df['datetime'], 0, pnl_df['cumulative_pnl'], 
                            where=(pnl_df['cumulative_pnl'] >= 0), alpha=0.3, color='green', label='盈利区间')
            ax2.fill_between(pnl_df['datetime'], 0, pnl_df['cumulative_pnl'], 
                            where=(pnl_df['cumulative_pnl'] < 0), alpha=0.3, color='red', label='亏损区间')
        
        ax2.set_title('累计盈亏趋势', fontsize=12, fontweight='bold')
        ax2.set_ylabel('累计盈亏', fontsize=10)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 持仓层级变化
        ax3 = axes[2]
        
        # 使用记录的层级历史
        if system.level_history:
            level_df = pd.DataFrame(system.level_history)
            level_df['datetime'] = pd.to_datetime(level_df['timestamp'], unit='s')
            level_df = level_df.sort_values('datetime')
            
            ax3.plot(level_df['datetime'], level_df['buy_level'], label='买单层级', linewidth=1.5, color='green', alpha=0.7)
            ax3.plot(level_df['datetime'], level_df['sell_level'], label='卖单层级', linewidth=1.5, color='red', alpha=0.7)
            ax3.fill_between(level_df['datetime'], 0, level_df['buy_level'], alpha=0.2, color='green')
            ax3.fill_between(level_df['datetime'], 0, level_df['sell_level'], alpha=0.2, color='red')
        
        ax3.set_title('持仓层级变化', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间', fontsize=10)
        ax3.set_ylabel('层级', fontsize=10)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'martin_grid_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        plt.close()
    
    # 调用绘图函数
    try:
        plot_martin_grid_results(df, system, results_dir)
    except Exception as e:
        print(f"绘图时出错: {e}")
        import traceback
        traceback.print_exc()

