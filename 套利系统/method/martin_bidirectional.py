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

数据来源：多交易所
- 支持 OKX / Binance 历史 K 线按日期下载
- 支持获取实时价格与实时 K 线
- 支持回测功能
"""

import pandas as pd
import numpy as np
import json
import os
import datetime
import matplotlib.pyplot as plt
from glob import glob
import csv
import requests
import time
from typing import Optional, List, Dict
import sys

# 添加父目录到路径，以便导入config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 全项目统一使用北京时间（东八区），不再使用其他时区
BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))


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
                 grid_spacing_mode='pct',      # 网格间距模式：'pct'(百分比), 'fixed'(固定点数), 'atr'(ATR动态)
                 grid_spacing_pct=0.01,        # 网格间距（百分比，mode='pct'时使用，如0.01表示1%）
                 grid_spacing_fixed=None,      # 网格间距（固定价格点数，mode='fixed'时使用，如100表示每层100 USDT）
                 atr_period=14,                # ATR周期（mode='atr'时用于计算ATR）
                 atr_multiplier=1.0,           # ATR倍数（mode='atr'时 间距=atr_multiplier*ATR）
                 base_size=0.01,               # 初始手数
                 multiplier=2.0,               # 马丁倍数（通常1.6~3.0）
                 max_martin_levels=8,          # 最大马丁层数（通常4~10）
                 normal_levels=3,              # 普通层数（前N层使用固定手数）
                 max_position_pct=0.9,         # 总最大仓位限制（资金百分比）
                 take_profit_pct=0.005,       # 止盈百分比（小利润，unified/per_trade 用）
                 take_profit_pct_by_level=None,  # 分层止盈时各层级止盈比例，如 [0.005,0.006,0.007,...]，None 则按 take_profit_pct + 层级索引 * 0.001 递增
                 stop_loss_pct=0.1,           # 止损百分比（总浮亏预警线）
                 take_profit_mode='unified',   # 止盈方式：'unified'(统一回本), 'per_trade'(逐笔), 'layered'(分层)
                 dynamic_base=True,            # 是否动态调整基准价（平仓后重置为当前价格）
                 total_capital=10000,          # 总资金（用于仓位限制计算）
                 fee_rate=0.0005):                # 回测交易费率（如 0.0005 表示 0.05%，按成交额收取）
        """
        初始化双向马丁网格系统
        
        Args:
            base_price: 基准价格，None表示使用第一个价格
            grid_spacing_mode: 网格间距模式 'pct'|'fixed'|'atr'
            grid_spacing_pct: 网格间距百分比（mode='pct' 时使用）
            grid_spacing_fixed: 网格间距固定点数（mode='fixed' 时使用）
            atr_period: ATR 周期（mode='atr' 时使用）
            atr_multiplier: ATR 倍数（mode='atr' 时 间距=atr_multiplier*ATR）
            base_size: 初始手数
            multiplier: 马丁倍数
            max_martin_levels: 最大马丁层数
            normal_levels: 普通层数（前N层使用固定手数）
            max_position_pct: 最大仓位限制（资金百分比）
            take_profit_pct: 止盈百分比（unified/per_trade 用）
            take_profit_pct_by_level: 分层止盈时各层级止盈比例列表，None 则自动按 take_profit_pct + 层级索引 * 0.001 递增
            stop_loss_pct: 止损百分比
            take_profit_mode: 止盈模式
            dynamic_base: 是否动态调整基准价
            total_capital: 总资金
            fee_rate: 回测交易费率（按成交额比例，如 0.0005 = 0.05%）
        """
        # 参数设置
        self.grid_spacing_mode = (grid_spacing_mode or 'pct').lower().strip()
        if self.grid_spacing_mode not in ('pct', 'fixed', 'atr'):
            self.grid_spacing_mode = 'pct'
        self.grid_spacing_pct = grid_spacing_pct
        self.grid_spacing_fixed = grid_spacing_fixed if grid_spacing_fixed is not None else 100.0  # mode='fixed' 时默认 100 价格单位
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self._last_atr = None  # 用于 ATR 模式下平仓后无新 atr 时沿用
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
        self.fee_rate = max(0.0, float(fee_rate))
        total_levels = self.normal_levels + self.max_martin_levels
        if take_profit_pct_by_level is not None:
            self.take_profit_pct_by_level = list(take_profit_pct_by_level)
            if len(self.take_profit_pct_by_level) < total_levels:
                last = self.take_profit_pct_by_level[-1] if self.take_profit_pct_by_level else take_profit_pct
                self.take_profit_pct_by_level.extend([last] * (total_levels - len(self.take_profit_pct_by_level)))
        else:
            # 按层级递增：第1层 take_profit_pct，第2层 +0.001，第3层 +0.002，...
            self.take_profit_pct_by_level = [take_profit_pct + i * 0.001 for i in range(total_levels)]
   
        # 状态变量
        self.base_price = base_price
        self.buy_level = 0          # 买单当前层级
        self.sell_level = 0          # 卖单当前层级
        
        # 持仓记录
        self.buy_positions = []      # 买单持仓列表 [{price, size, level, timestamp}]
        self.sell_positions = []      # 卖单持仓列表
        
        # 交易记录
        self.trade_history = []      # 所有交易记录
        self.total_profit = 0.0      # 累计已实现盈亏（已扣除手续费）
        self.total_fees = 0.0        # 累计交易手续费（回测用）
        self.close_history = []      # 平仓历史记录
        
        # 网格线
        self.buy_grid_lines = []     # 上方网格线（买单触发线）
        self.sell_grid_lines = []    # 下方网格线（卖单触发线）
        
        # 状态历史（用于绘图）
        self.state_history = []      # [{timestamp, price, buy_level, sell_level, unrealized_pnl}]
        # 仓位限制跳过告警：同一 (方向, 层级) 只告警一次，避免刷屏（行为与实盘一致：超限则不下单）
        self._position_limit_skip_warned = set()  # 元素为 ("buy", level) 或 ("sell", level)
    
    def initialize_base_price(self, current_price, atr=None):
        """初始化基准价格；mode='atr' 时可传入 atr 用于生成网格线"""
        if self.base_price is None:
            self.base_price = current_price
            self._update_grid_lines(atr=atr)
            spacing = self._get_grid_spacing(atr=atr)
            print(f"初始化基准价格: {self.base_price:.2f}, 间距模式={self.grid_spacing_mode}, 当前间距={spacing:.4f}")
    
    def _get_grid_spacing(self, atr=None):
        """
        根据当前间距模式计算单层网格间距。
        - pct: 基准价 * grid_spacing_pct
        - fixed: grid_spacing_fixed（固定价格点数）
        - atr: atr_multiplier * ATR（传入或沿用 _last_atr）
        """
        if self.grid_spacing_mode == 'pct':
            return self.base_price * self.grid_spacing_pct
        if self.grid_spacing_mode == 'fixed':
            return self.grid_spacing_fixed
        # atr
        a = atr if atr is not None and atr > 0 else self._last_atr
        if a is not None and a > 0:
            return self.atr_multiplier * a
        # 无有效 ATR 时回退为基准价的 1%
        return self.base_price * 0.01
    
    def _update_grid_lines(self, atr=None):
        """
        更新网格线（基准价格上下）
        
        支持三种间距模式（见 PDF）：固定点数、百分比、ATR 动态。
        在基准价格上方和下方生成等间距的网格线：
        - 上方网格线：用于触发买单（价格上涨时）
        - 下方网格线：用于触发卖单（价格下跌时）
        """
        if self.base_price is None:
            return
        if self.grid_spacing_mode == 'atr' and atr is not None and atr > 0:
            self._last_atr = atr
        
        spacing = self._get_grid_spacing(atr=atr)
        total_levels = self.normal_levels + self.max_martin_levels
        
        # 生成上方网格线（买单触发线）：base_price + spacing * level
        self.buy_grid_lines = [
            self.base_price + spacing * level
            for level in range(1, total_levels + 1)
        ]
        # 生成下方网格线（卖单触发线）：base_price - spacing * level
        self.sell_grid_lines = [
            self.base_price - spacing * level
            for level in range(1, total_levels + 1)
        ]
    
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
        
        # 检查仓位限制（超限则不下单，与实盘一致：实盘也会因自设或交易所限制而不成交）
        if not self._check_position_limit(new_position_value):
            key = ('buy', level)
            if key not in self._position_limit_skip_warned:
                self._position_limit_skip_warned.add(key)
                print(f"警告：开买单超过仓位限制，跳过 (level={level}, size={size})，后续同层级不再重复提示")
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
        
        # 回测手续费：按成交额收取
        fee = (size * price) * self.fee_rate
        self.total_fees += fee
        self.total_profit -= fee
        
        # 记录交易
        rec = {
            'action': 'open_buy',
            'price': price,
            'size': size,
            'level': level,
            'timestamp': timestamp,
            'grid_line': self.buy_grid_lines[level - 1] if level <= len(self.buy_grid_lines) else None
        }
        if fee > 0:
            rec['fee'] = fee
        self.trade_history.append(rec)
        
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
        
        # 检查仓位限制（超限则不下单，与实盘一致：实盘也会因自设或交易所限制而不成交）
        if not self._check_position_limit(new_position_value):
            key = ('sell', level)
            if key not in self._position_limit_skip_warned:
                self._position_limit_skip_warned.add(key)
                print(f"警告：开卖单超过仓位限制，跳过 (level={level}, size={size})，后续同层级不再重复提示")
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
        
        # 回测手续费：按成交额收取
        fee = (size * price) * self.fee_rate
        self.total_fees += fee
        self.total_profit -= fee
        
        # 记录交易
        rec = {
            'action': 'open_sell',
            'price': price,
            'size': size,
            'level': level,
            'timestamp': timestamp,
            'grid_line': self.sell_grid_lines[level - 1] if level <= len(self.sell_grid_lines) else None
        }
        if fee > 0:
            rec['fee'] = fee
        self.trade_history.append(rec)
        
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
            # 3. 分层止盈：根据当前层级使用对应止盈比例（层级越高，止盈比例可设越高）
            if len(self.buy_positions) > 0:
                buy_avg = self._calculate_average_price(self.buy_positions)
                if buy_avg:
                    buy_pnl_pct = (current_price - buy_avg) / buy_avg
                    required_buy = self._get_take_profit_pct_for_level(self.buy_level)
                    if buy_pnl_pct >= required_buy:
                        return True, 'buy_profit'
            
            if len(self.sell_positions) > 0:
                sell_avg = self._calculate_average_price(self.sell_positions)
                if sell_avg:
                    sell_pnl_pct = (sell_avg - current_price) / sell_avg
                    required_sell = self._get_take_profit_pct_for_level(self.sell_level)
                    if sell_pnl_pct >= required_sell:
                        return True, 'sell_profit'
        
        return False, None
    
    def _get_take_profit_pct_for_level(self, level):
        """分层止盈：根据当前层级返回该层级对应的止盈比例。level 从 1 开始。"""
        if level <= 0:
            return self.take_profit_pct_by_level[0] if self.take_profit_pct_by_level else self.take_profit_pct
        idx = min(level - 1, len(self.take_profit_pct_by_level) - 1)
        return self.take_profit_pct_by_level[idx]
    
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
        
        # 平仓手续费：按平仓成交额（当前价 × 数量）收取
        close_notional = sum(p['size'] * current_price for p in self.buy_positions + self.sell_positions)
        close_fee = close_notional * self.fee_rate
        self.total_fees += close_fee
        self.total_profit -= close_fee
        
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
            'close_fee': close_fee,
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'buy_avg_price': self._calculate_average_price(self.buy_positions),
            'sell_avg_price': self._calculate_average_price(self.sell_positions)
        }
        
        self.close_history.append(close_info)
        
        # 记录交易历史
        rec = {
            'action': 'close_all',
            'price': current_price,
            'timestamp': timestamp,
            'reason': reason,
            'realized_pnl': realized_pnl,
            'buy_count': len(self.buy_positions),
            'sell_count': len(self.sell_positions)
        }
        if close_fee > 0:
            rec['fee'] = close_fee
        self.trade_history.append(rec)
        
        # 清空持仓
        self.buy_positions = []
        self.sell_positions = []
        
        # 重置层级
        self.buy_level = 0
        self.sell_level = 0
        # 新一轮允许再次提示仓位限制跳过（同一层级在新周期可能再次超限）
        self._position_limit_skip_warned.clear()
        
        # 如果启用动态基准，更新基准价
        if self.dynamic_base:
            old_base = self.base_price
            self.base_price = current_price
            self._update_grid_lines()
            print(f"平仓后更新基准价: {old_base:.2f} → {self.base_price:.2f} (原因: {reason})")
        
        return close_info
    
    def process_price(self, current_price, timestamp, atr=None):
        """
        处理当前价格（主循环逻辑）
        
        流程：
        1. 初始化基准价格（如果未初始化）
        2. ATR 模式下用当前 atr 更新网格线
        3. 检查止损
        4. 检查止盈
        5. 检查网格线触发（上方/下方）
        6. 记录状态历史
        
        atr: 当前 ATR 值，仅当 grid_spacing_mode='atr' 时使用（回测时由外部计算并传入）
        """
        # 初始化基准价格
        if self.base_price is None:
            self.initialize_base_price(current_price, atr=atr)
        # ATR 模式：每根 K 线用当前 ATR 更新网格线（动态间距）
        elif self.grid_spacing_mode == 'atr' and atr is not None and atr > 0:
            self._update_grid_lines(atr=atr)
        
        # 计算未实现盈亏
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        
        # 记录状态历史（包含基准价）
        self.state_history.append({
            'timestamp': timestamp,
            'price': current_price,
            'base_price': self.base_price,  # 记录当前基准价
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
            'total_fees': self.total_fees,
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


# ========== 数据下载函数 ==========

def download_okx_klines(symbol: str, interval: str, start_time: datetime.datetime,
                       end_time: datetime.datetime, output_dir: str,
                       proxies: Optional[Dict] = None) -> List[str]:
    """
    下载OKX历史K线数据
    
    Args:
        symbol: 交易对，如 'BTC-USDT-SWAP'
        interval: 时间间隔，如 '1m', '5m', '1H', '1D' (OKX使用大写H和D)
        start_time: 开始时间
        end_time: 结束时间
        output_dir: 输出目录
        proxies: 代理设置（可选）
    
    Returns:
        下载的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # OKX API限制：每次最多100条
    limit = 100
    # 转换interval格式（OKX需要大写H和D）
    okx_interval_map = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1H',
        '1H': '1H',
        '4h': '4H',
        '4H': '4H',
        '1d': '1D',
        '1D': '1D'
    }
    okx_interval = okx_interval_map.get(interval, interval)
    
    interval_seconds_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1H': 3600,
        '4H': 14400,
        '1D': 86400
    }
    interval_seconds = interval_seconds_map.get(okx_interval, 60)
    
    # 将北京时间转换为 UTC 时间戳（毫秒），供 API 使用
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=BEIJING_TZ)
        end_time = end_time.replace(tzinfo=BEIJING_TZ)
    start_time_utc = start_time.astimezone(datetime.timezone.utc)
    end_time_utc = end_time.astimezone(datetime.timezone.utc)
    
    start_ms = int(start_time_utc.timestamp() * 1000)
    end_ms = int(end_time_utc.timestamp() * 1000)
    
    downloaded_files = []
    written = set()  # (filename, open_time_ms) 本轮已写入，避免重复 K 线
    cursor = start_ms
    
    print(f"开始下载OKX {symbol} {interval} K线数据...")
    print(f"时间范围 (北京时间): {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"时间戳范围: {start_ms} 至 {end_ms}")
    
    url = "https://www.okx.com/api/v5/market/history-candles"
    
    while cursor < end_ms:
        params = {
            'instId': symbol,
            'bar': okx_interval,
            'after': str(cursor),  # 返回该时间戳之后的 K 线（从前往后拉）
            'limit': str(limit)
        }
        
        try:
            response = requests.get(url, params=params, timeout=30, proxies=proxies)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') != '0' or not data.get('data'):
                break
            
            klines = data['data']
            
            if not klines:
                break
            
            valid_klines = []
            step_ms = interval_seconds * 1000
            for kline in klines:
                open_time_ms = int(kline[0])
                if open_time_ms < start_ms or open_time_ms > end_ms:
                    continue
                kline_time_beijing = datetime.datetime.fromtimestamp(open_time_ms / 1000, tz=datetime.timezone.utc).astimezone(BEIJING_TZ)
                date_str = kline_time_beijing.strftime('%Y-%m-%d')
                filename = os.path.join(output_dir, f"{symbol.replace('-', '')}-{okx_interval}-{date_str}.csv")
                key = (filename, open_time_ms)
                if key in written:
                    continue
                written.add(key)
                valid_klines.append(kline)
                file_exists = os.path.exists(filename)
                mode = 'a' if file_exists else 'w'
                with open(filename, mode, newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    row = [
                        open_time_ms * 1000,
                        kline[1], kline[2], kline[3], kline[4], kline[5],
                        open_time_ms * 1000, kline[6], 0, kline[5], kline[6], 0
                    ]
                    writer.writerow(row)
                if filename not in downloaded_files:
                    downloaded_files.append(filename)
            
            # 根据本批数据推进：有数据则用本批最大时间戳；无进展则按一批量（limit 根）大步进，减少请求次数
            first_ts = int(klines[0][0])
            last_ts = int(klines[-1][0])
            newest_time_ms = max(first_ts, last_ts)
            if newest_time_ms >= end_ms or len(klines) < limit:
                break
            next_cursor = (newest_time_ms // step_ms + 1) * step_ms
            if newest_time_ms < start_ms:
                next_cursor = start_ms
            if next_cursor <= cursor:
                next_cursor = cursor + limit * step_ms
            if next_cursor > end_ms:
                break
            cursor = next_cursor
            
            try:
                next_str = datetime.datetime.fromtimestamp(cursor / 1000, tz=datetime.timezone.utc).astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                next_str = str(cursor)
            print(f"  已下载: {len(valid_klines)} 条有效数据（本次获取 {len(klines)} 条），下一批 after: {next_str} 北京时间")
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"下载数据时出错: {e}")
            break
    
    downloaded_files = sorted(list(set(downloaded_files)))
    print(f"下载完成！共 {len(downloaded_files)} 个文件")
    return downloaded_files


def download_binance_klines(symbol: str, interval: str, start_time: datetime.datetime,
                            end_time: datetime.datetime, output_dir: str,
                            use_futures: bool = False,
                            proxies: Optional[Dict] = None) -> List[str]:
    """
    下载 Binance 历史 K 线数据（现货或 U 本位合约）
    
    Args:
        symbol: 交易对，如 'BTCUSDT'（无横线）
        interval: 时间间隔，如 '1m', '5m', '15m', '1h', '4h', '1d'
        start_time: 开始时间
        end_time: 结束时间
        output_dir: 输出目录
        use_futures: True 使用 U 本位合约 API，False 使用现货 API
        proxies: 代理设置（可选）
    
    Returns:
        下载的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    # Binance 单次最多 1000 条
    limit = 1000
    # interval 统一为小写，Binance 接受 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    binance_interval = interval.lower()
    
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=BEIJING_TZ)
        end_time = end_time.replace(tzinfo=BEIJING_TZ)
    start_time_utc = start_time.astimezone(datetime.timezone.utc)
    end_time_utc = end_time.astimezone(datetime.timezone.utc)
    
    start_ms = int(start_time_utc.timestamp() * 1000)
    end_ms = int(end_time_utc.timestamp() * 1000)
    
    base_url = "https://fapi.binance.com/fapi/v1/klines" if use_futures else "https://api.binance.com/api/v3/klines"
    binance_symbol = symbol.replace('-', '').upper()
    downloaded_files = []
    current_start = start_ms
    _451_fallback_done = False
    
    # 代理诊断：确认请求会走代理
    if proxies and (proxies.get('https') or proxies.get('http')):
        _p = proxies.get('https') or proxies.get('http') or ''
        if '@' in _p:
            _p = _p.split('@')[-1]
        print(f"[代理] 使用: {_p} （请求将经此代理访问 Binance）")
        if _p.startswith('socks'):
            print("[代理] 使用 SOCKS 时需安装: pip install PySocks")
        elif not _p.startswith('http'):
            print("[代理] 提示: 若 451，可尝试 https 写为 http 代理地址，如 'https': 'http://127.0.0.1:7897'")
    else:
        print("[代理] 未配置 proxies，直连可能触发 451；可设置 data_config['proxies'] 或环境变量 HTTPS_PROXY")
    
    print(f"开始下载 Binance {'合约' if use_futures else '现货'} {binance_symbol} {interval} K线数据...")
    print(f"时间范围 (北京时间): {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    while current_start < end_ms:
        params = {
            'symbol': binance_symbol,
            'interval': binance_interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': limit
        }
        try:
            response = requests.get(base_url, params=params, timeout=30, proxies=proxies)
            if response.status_code == 451:
                if not use_futures and not _451_fallback_done:
                    _451_fallback_done = True
                    use_futures = True
                    base_url = "https://fapi.binance.com/fapi/v1/klines"
                    print("现货 API 返回 451，自动改用合约 API 重试...")
                    continue
                _hint = (
                    "Binance 返回 451（地区/法律限制）。开启代理后仍 451 的常见原因：\n"
                    "  ① 代理未生效：确认代理软件已开启、端口与 data_config['proxies'] 一致（如 7897）；"
                    " 运行时应看到 [代理] 使用: http://127.0.0.1:7897，若未看到或端口不对请修正。\n"
                    "  ② 代理出口 IP 被 Binance 限制：部分 VPN/代理节点会被拒绝，可更换节点或改用 exchange='okx' 回测。"
                )
                raise requests.exceptions.HTTPError(_hint, response=response)
            response.raise_for_status()
            klines = response.json()
            if not klines:
                break
            for k in klines:
                # Binance: [open_time, open, high, low, close, volume, close_time, quote_vol, trades, ...]
                open_time_ms = int(k[0])
                if open_time_ms < start_ms or open_time_ms > end_ms:
                    continue
                kline_time_beijing = datetime.datetime.fromtimestamp(open_time_ms / 1000, tz=datetime.timezone.utc).astimezone(BEIJING_TZ)
                date_str = kline_time_beijing.strftime('%Y-%m-%d')
                filename = os.path.join(output_dir, f"{binance_symbol}-{binance_interval}-{date_str}.csv")
                file_exists = os.path.exists(filename)
                mode = 'a' if file_exists else 'w'
                with open(filename, mode, newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    row = [
                        open_time_ms * 1000,
                        k[1], k[2], k[3], k[4], k[5],
                        int(k[6]) * 1000,
                        k[7], int(k[8]) if len(k) > 8 else 0,
                        k[9] if len(k) > 9 else k[5], k[10] if len(k) > 10 else k[7],
                        0
                    ]
                    writer.writerow(row)
                if filename not in downloaded_files:
                    downloaded_files.append(filename)
            last_time = int(klines[-1][0])
            if len(klines) < limit:
                break
            current_start = last_time + 1
            print(f"  已获取至 {datetime.datetime.fromtimestamp(last_time/1000, tz=datetime.timezone.utc).astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')} 北京时间")
            time.sleep(0.2)
        except Exception as e:
            print(f"下载 Binance 数据时出错: {e}")
            break
    
    downloaded_files = sorted(list(set(downloaded_files)))
    print(f"Binance 下载完成！共 {len(downloaded_files)} 个文件")
    return downloaded_files


def ensure_data_downloaded(exchange: str, symbol: str, interval: str,
                          start_date: str, end_date: str,
                          data_dir: Optional[str] = None,
                          proxies: Optional[Dict] = None,
                          binance_futures: bool = False) -> List[str]:
    """
    确保所需数据已下载，若缺失则按日期范围自动下载。
    支持交易所：okx / binance
    
    Args:
        exchange: 交易所，'okx' 或 'binance'
        symbol: 交易对。OKX 如 'BTC-USDT-SWAP'，Binance 如 'BTCUSDT'
        interval: 时间间隔，如 '1m', '5m', '15m', '1h', '4h', '1d'
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        data_dir: 数据根目录，None 则使用 config.DATA_DIR
        proxies: 代理，如 {'http':'...','https':'...'}
        binance_futures: 仅 Binance 有效，True 使用 U 本位合约，False 使用现货
    
    Returns:
        该交易所、交易对、周期下在日期范围内的数据文件路径列表（已排序）
    """
    ex = exchange.lower().strip()
    if ex not in ('okx', 'binance'):
        raise ValueError(f"不支持的交易所: {exchange}，仅支持 'okx' 或 'binance'")
    
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    # 日期按北京时间解析
    start_time = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=BEIJING_TZ)
    end_time = (datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)).replace(tzinfo=BEIJING_TZ)
    
    if ex == 'okx':
        output_dir = os.path.join(data_dir, f"OKX_{interval}_kline")
        okx_interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '1H': '1H', '4h': '4H', '4H': '4H', '1d': '1D', '1D': '1D'}
        okx_interval = okx_interval_map.get(interval, interval)
        if '-SWAP' in symbol.upper():
            okx_symbol = symbol
        else:
            base = symbol.replace('USDT', '').replace('usdt', '').replace('-', '').strip() or symbol.split('-')[0] or symbol
            okx_symbol = f"{base}-USDT-SWAP"
        file_prefix = okx_symbol.replace('-', '')
        pattern = os.path.join(output_dir, f"{file_prefix}-{okx_interval}-*.csv")
        needed_dates = []
        current = start_time
        while current <= end_time:
            date_str = current.strftime('%Y-%m-%d')
            fn = os.path.join(output_dir, f"{file_prefix}-{okx_interval}-{date_str}.csv")
            if not os.path.exists(fn):
                needed_dates.append(current)
            current += datetime.timedelta(days=1)
        if needed_dates:
            print(f"需要下载 {len(needed_dates)} 天的 OKX 数据")
            download_okx_klines(okx_symbol, interval, start_time, end_time, output_dir, proxies)
        else:
            print(f"OKX 数据已存在，跳过下载")
        return sorted(glob(pattern))
    
    else:
        # binance
        output_dir = os.path.join(data_dir, f"Binance_{interval}_kline")
        binance_interval = interval.lower()
        binance_symbol = symbol.replace('-', '').upper()
        pattern = os.path.join(output_dir, f"{binance_symbol}-{binance_interval}-*.csv")
        needed_dates = []
        current = start_time
        while current <= end_time:
            date_str = current.strftime('%Y-%m-%d')
            fn = os.path.join(output_dir, f"{binance_symbol}-{binance_interval}-{date_str}.csv")
            if not os.path.exists(fn):
                needed_dates.append(current)
            current += datetime.timedelta(days=1)
        if needed_dates:
            print(f"需要下载 {len(needed_dates)} 天的 Binance 数据")
            got = download_binance_klines(binance_symbol, interval, start_time, end_time, output_dir, use_futures=binance_futures, proxies=proxies)
            if len(got) == 0:
                print("警告：Binance 未下载到新数据（多为 451 代理出口 IP 被限制）。若目录内已有文件将沿用，建议改用 exchange='okx' 或更换代理节点后再试。")
        else:
            print(f"Binance 数据已存在，跳过下载")
        return sorted(glob(pattern))


def get_realtime_price(exchange: str, symbol: str,
                       proxies: Optional[Dict] = None) -> Dict:
    """
    获取指定交易所、交易对的实时最新价。
    
    Args:
        exchange: 'okx' 或 'binance'
        symbol: OKX 如 'BTC-USDT-SWAP'，Binance 如 'BTCUSDT'
        proxies: 代理（可选）
    
    Returns:
        {'price': float, 'timestamp': int 秒, 'exchange': str}
        失败时 price 为 0.0，timestamp 为 0
    """
    ex = exchange.lower().strip()
    result = {'price': 0.0, 'timestamp': 0, 'exchange': ex}
    try:
        if ex == 'okx':
            inst_id = symbol if '-SWAP' in symbol else f"{symbol.replace('USDT', '').replace('usdt', '')}-USDT-SWAP"
            url = f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}"
            r = requests.get(url, timeout=10, proxies=proxies)
            r.raise_for_status()
            data = r.json()
            if data.get('code') == '0' and data.get('data'):
                result['price'] = float(data['data'][0]['last'])
                result['timestamp'] = int(int(data['data'][0]['ts']) / 1000)
        elif ex == 'binance':
            sym = symbol.replace('-', '').upper()
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={sym}"
            r = requests.get(url, timeout=10, proxies=proxies)
            r.raise_for_status()
            data = r.json()
            result['price'] = float(data.get('price', 0))
            result['timestamp'] = int(time.time())
        else:
            result['price'] = 0.0
            result['timestamp'] = 0
    except Exception as e:
        print(f"获取实时价格失败 ({exchange} {symbol}): {e}")
    return result


def get_realtime_candle(exchange: str, symbol: str, interval: str,
                        use_futures: bool = False,
                        proxies: Optional[Dict] = None) -> Dict:
    """
    获取当前/最近一根 K 线（实时）。
    
    Args:
        exchange: 'okx' 或 'binance'
        symbol: 交易对
        interval: 如 '1m', '5m', '1h'
        use_futures: 仅 Binance 有效，True 使用合约
        proxies: 代理（可选）
    
    Returns:
        {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float,
         'open_time': int ms, 'close_time': int ms, 'timestamp': int 秒}
        缺失时字段为 0
    """
    ex = exchange.lower().strip()
    out = {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0, 'open_time': 0, 'close_time': 0, 'timestamp': 0}
    try:
        if ex == 'okx':
            inst_id = symbol if '-SWAP' in symbol else f"{symbol.replace('USDT', '').replace('usdt', '')}-USDT-SWAP"
            okx_bar = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D'}.get(interval.lower(), interval)
            url = "https://www.okx.com/api/v5/market/candles"
            params = {'instId': inst_id, 'bar': okx_bar, 'limit': '1'}
            r = requests.get(url, params=params, timeout=10, proxies=proxies)
            r.raise_for_status()
            data = r.json()
            if data.get('code') == '0' and data.get('data'):
                k = data['data'][0]
                out['open_time'] = int(k[0])
                out['open'] = float(k[1]); out['high'] = float(k[2]); out['low'] = float(k[3]); out['close'] = float(k[4]); out['volume'] = float(k[5])
                out['close_time'] = out['open_time']
                out['timestamp'] = out['open_time'] // 1000
        elif ex == 'binance':
            sym = symbol.replace('-', '').upper()
            base = "https://fapi.binance.com/fapi/v1/klines" if use_futures else "https://api.binance.com/api/v3/klines"
            params = {'symbol': sym, 'interval': interval.lower(), 'limit': 1}
            r = requests.get(base, params=params, timeout=10, proxies=proxies)
            r.raise_for_status()
            arr = r.json()
            if arr:
                k = arr[-1]
                out['open_time'] = int(k[0]); out['close_time'] = int(k[6])
                out['open'] = float(k[1]); out['high'] = float(k[2]); out['low'] = float(k[3]); out['close'] = float(k[4]); out['volume'] = float(k[5])
                out['timestamp'] = out['open_time'] // 1000
    except Exception as e:
        print(f"获取实时K线失败 ({exchange} {symbol}): {e}")
    return out


# ========== ATR 计算（用于网格间距 atr 模式）==========

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算 Wilder 平滑 ATR（Average True Range）。
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    首根 ATR = 前 period 根 K 线的 TR 的均值，之后 ATR = (prev_ATR*(period-1) + TR) / period
    """
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ========== 数据加载函数 ==========

def load_price_data(filename):
    """
    加载 K 线价格数据（多交易所统一格式）
    
    支持格式：
    1. 无表头 CSV（由 download_okx_klines / download_binance_klines 生成）：
       时间戳(微秒),开盘价,最高价,最低价,收盘价,成交量,...
    2. 有表头 CSV：包含 open_time 和 close 字段
    """
    prices = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # 读取第一行判断格式
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            # 判断是否为无表头格式（纯数字，用逗号分隔，第一列是时间戳）
            if ',' in first_line and first_line.split(',')[0].isdigit():
                # 无表头格式：时间戳(微秒),开盘价,最高价,最低价,收盘价,成交量,...
                reader = csv.reader(f)
                for row in reader:
                    try:
                        if len(row) < 5:
                            continue
                        ts_microseconds = int(row[0])
                        ts = ts_microseconds // 1000000  # 转换为秒
                        o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                        if ts > 0 and c > 0:
                            prices.append({'timestamp': ts, 'price': c, 'open': o, 'high': h, 'low': l, 'close': c})
                    except (ValueError, IndexError):
                        continue
            else:
                # 有表头格式：尝试读取DictReader
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = int(row.get('open_time', row.get('timestamp', 0))) // 1000
                        price = float(row.get('close', row.get('price', 0)))
                        if ts > 0 and price > 0:
                            rec = {'timestamp': ts, 'price': price}
                            for k in ('open', 'high', 'low'):
                                if k in row and row[k]:
                                    rec[k] = float(row[k])
                            if 'open' in rec and 'high' in rec and 'low' in rec:
                                rec['close'] = price
                            prices.append(rec)
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        print(f"加载文件 {filename} 时出错: {e}")
    
    return prices


# ========== 主程序 ==========

def _default_data_config():
    """默认数据配置：来自 config.MARTIN_DATA_CONFIG，并补全 data_dir。"""
    out = dict(getattr(config, 'MARTIN_DATA_CONFIG', {}))
    out.setdefault('exchange', 'okx')
    out.setdefault('symbol', 'BTC-USDT-SWAP')
    out.setdefault('interval', '1m')
    out.setdefault('start_date', '2026-02-01')
    out.setdefault('end_date', '2026-02-02')
    out.setdefault('binance_futures', False)
    out.setdefault('proxies', None)
    out['data_dir'] = getattr(config, 'DATA_DIR', out.get('data_dir'))
    return out


def _default_strategy_config():
    """默认策略配置：来自 config.MARTIN_STRATEGY_CONFIG。"""
    out = dict(getattr(config, 'MARTIN_STRATEGY_CONFIG', {}))
    out.setdefault('base_price', None)
    out.setdefault('grid_spacing_mode', 'atr')
    out.setdefault('grid_spacing_pct', 0.01)
    out.setdefault('grid_spacing_fixed', 100.0)
    out.setdefault('atr_period', 14)
    out.setdefault('atr_multiplier', 1.0)
    out.setdefault('base_size', 0.01)
    out.setdefault('multiplier', 2.0)
    out.setdefault('max_martin_levels', 8)
    out.setdefault('normal_levels', 3)
    out.setdefault('max_position_pct', 0.9)
    out.setdefault('take_profit_pct', 0.005)
    out.setdefault('stop_loss_pct', 0.1)
    out.setdefault('take_profit_mode', 'unified')
    out.setdefault('dynamic_base', True)
    out.setdefault('total_capital', 10000)
    out.setdefault('fee_rate', 0.0005)
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='双向马丁网格策略：回测/实时')
    parser.add_argument('--config', type=str, default=None, help='JSON 配置文件路径，含 data_config 与 strategy_config')
    args = parser.parse_args()

    run_mode = getattr(config, 'RUN_MODE', 'backtest')
    poll_interval_seconds = getattr(config, 'POLL_INTERVAL_SECONDS', 60)

    if args.config and os.path.isfile(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            data_config = file_config.get('data_config', _default_data_config())
            strategy_config = file_config.get('strategy_config', _default_strategy_config())
            run_mode = file_config.get('run_mode', run_mode)
            poll_interval_seconds = file_config.get('poll_interval_seconds', poll_interval_seconds)
            if data_config.get('data_dir') is None:
                data_config['data_dir'] = config.DATA_DIR
            if data_config.get('proxies') is None:
                data_config['proxies'] = None
            strategy_config['base_price'] = strategy_config.get('base_price')  # JSON 里通常为 null
            print(f"已从配置文件加载: {args.config}")
        except Exception as e:
            print(f"读取配置失败: {e}，使用默认配置")
            data_config = _default_data_config()
            strategy_config = _default_strategy_config()
    else:
        data_config = _default_data_config()
        strategy_config = _default_strategy_config()

    # 合并配置（用于保存结果），不覆盖 config 模块以便使用 config.RESULTS_DIR 等
    run_config = {**data_config, **strategy_config}
    
    print("=" * 60)
    print("双向马丁网格策略系统")
    print("=" * 60)
    print(f"数据配置:")
    for key, value in data_config.items():
        if key != 'proxies':
            print(f"  {key}: {value}")
    print(f"\n策略参数配置:")
    for key, value in strategy_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    run_mode = (run_mode or 'backtest').strip().lower()
    if run_mode not in ('backtest', 'realtime'):
        run_mode = 'backtest'
    print(f"运行模式: {run_mode}")
    
    if run_mode == 'backtest':
        # ===== 自动下载数据 =====
        print("\n检查并下载数据...")
        try:
            price_files = ensure_data_downloaded(
                exchange=data_config['exchange'],
                symbol=data_config['symbol'],
                interval=data_config['interval'],
                start_date=data_config['start_date'],
                end_date=data_config['end_date'],
                data_dir=data_config.get('data_dir'),
                proxies=data_config.get('proxies'),
                binance_futures=data_config.get('binance_futures', False)
            )
            if not price_files:
                print("错误：未能找到或下载任何数据文件")
                exit(1)
            print(f"找到 {len(price_files)} 个数据文件")
        except Exception as e:
            print(f"数据下载/加载失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        
        # ===== 加载数据 =====
        print("\n加载价格数据...")
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
        
        # 构建 DataFrame（保留 open/high/low/close 用于 ATR 模式）
        df = pd.DataFrame(all_prices)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df = df.drop_duplicates()
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            atr_period = strategy_config.get('atr_period', 14)
            df['atr'] = compute_atr(df['high'].values, df['low'].values, df['close'].values, atr_period)
        else:
            df['atr'] = np.nan
        start_dt = datetime.datetime.strptime(data_config['start_date'], '%Y-%m-%d').replace(tzinfo=BEIJING_TZ)
        end_dt = (datetime.datetime.strptime(data_config['end_date'], '%Y-%m-%d') + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)).replace(tzinfo=BEIJING_TZ)
        start_ts_sec = int(start_dt.timestamp())
        end_ts_sec = int(end_dt.timestamp())
        mask = (df.index >= start_ts_sec) & (df.index <= end_ts_sec)
        df = df.loc[mask]
        dt_utc = pd.to_datetime(df.index, unit='s', utc=True)
        df['datetime'] = dt_utc.tz_convert('Asia/Shanghai').tz_localize(None)
        df = df.dropna(subset=['price'])
        if df.empty:
            print("筛选后的数据为空")
            exit(1)
        print(f"加载了 {len(df)} 条价格数据")
        print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        print(f"价格范围: {df['price'].min():.2f} 至 {df['price'].max():.2f}")
        
        # ===== 初始化系统 =====
        print("\n初始化双向马丁网格系统...")
        system = BidirectionalMartinGrid(**strategy_config)
        # ===== 回测 =====
        print("\n开始回测...")
        use_atr = strategy_config.get('grid_spacing_mode') == 'atr'
        for idx, row in df.iterrows():
            atr_val = row.get('atr') if use_atr and 'atr' in row.index else None
            if atr_val is not None and (np.isnan(atr_val) or atr_val <= 0):
                atr_val = None
            system.process_price(row['price'], idx, atr=atr_val)
        if system.buy_positions or system.sell_positions:
            final_price = df.iloc[-1]['price']
            final_timestamp = df.index[-1]
            system.close_all_positions(final_price, final_timestamp, 'end_of_data')
        
        # ===== 保存结果 =====
        results_dir = config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'martin_bidirectional_trades.json'), 'w', encoding='utf-8') as f:
            json.dump(system.trade_history, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(results_dir, 'martin_bidirectional_closes.json'), 'w', encoding='utf-8') as f:
            json.dump(system.close_history, f, ensure_ascii=False, indent=2, default=str)
        buy_trades = len([t for t in system.trade_history if t['action'] == 'open_buy'])
        sell_trades = len([t for t in system.trade_history if t['action'] == 'open_sell'])
        close_trades = len([t for t in system.trade_history if t['action'] == 'close_all'])
        results_summary = {
            '策略类型': '双向马丁网格',
            '最终盈亏': system.total_profit,
            '累计手续费': system.total_fees,
            '总交易次数': len(system.trade_history),
            '开买单次数': buy_trades,
            '开卖单次数': sell_trades,
            '平仓次数': close_trades,
            '数据配置': {k: v for k, v in data_config.items() if k != 'proxies'},
            '策略参数': strategy_config,
            '回测时间': datetime.datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            '数据时间范围': {'开始': str(df['datetime'].min()), '结束': str(df['datetime'].max())}
        }
        with open(os.path.join(results_dir, 'martin_bidirectional_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print("回测完成！")
        print("=" * 60)
        print(f"最终盈亏（已扣费）: {system.total_profit:.4f} USDT")
        if system.total_fees > 0:
            print(f"累计手续费: {system.total_fees:.4f} USDT")
        print(f"总交易次数: {len(system.trade_history)}")
        print(f"开买单次数: {buy_trades}")
        print(f"开卖单次数: {sell_trades}")
        print(f"平仓次数: {close_trades}")
        print(f"结果已保存到: {results_dir}")
        print("=" * 60)
        
        # ===== 绘图与图表数据 =====
        # 时间统一为北京时间（naive）
        def _ts_to_beijing_dt(ts):
            return pd.to_datetime(ts, unit='s', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)

        def plot_martin_bidirectional_results(df, system, output_dir):
            """绘制双向马丁网格策略结果"""
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            ax1 = axes[0]
            ax1.plot(df['datetime'], df['price'], label='价格', linewidth=1.5, alpha=0.7, color='black')
            if system.state_history:
                state_df = pd.DataFrame(system.state_history)
                state_df['datetime'] = state_df['timestamp'].apply(_ts_to_beijing_dt)
                state_df = state_df.sort_values('datetime')
                base_price_df = state_df[state_df['base_price'].notna()].copy()
                if not base_price_df.empty:
                    ax1.plot(base_price_df['datetime'], base_price_df['base_price'],
                             color='orange', linestyle='--', linewidth=2, label='基准价', alpha=0.8)
            buy_trades = [t for t in system.trade_history if t['action'] == 'open_buy']
            sell_trades = [t for t in system.trade_history if t['action'] == 'open_sell']
            close_trades = [t for t in system.trade_history if t['action'] == 'close_all']
            if buy_trades:
                buy_times = [_ts_to_beijing_dt(t['timestamp']) for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=80, label='开买单', zorder=5, alpha=0.8)
            if sell_trades:
                sell_times = [_ts_to_beijing_dt(t['timestamp']) for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=80, label='开卖单', zorder=5, alpha=0.8)
            if close_trades:
                close_times = [_ts_to_beijing_dt(t['timestamp']) for t in close_trades]
                close_prices = [t['price'] for t in close_trades]
                ax1.scatter(close_times, close_prices, color='blue', marker='x', s=150, label='平仓', zorder=5, alpha=0.9, linewidths=2)
            ax1.set_title('价格趋势与交易点', fontsize=14, fontweight='bold')
            ax1.set_ylabel('价格', fontsize=11)
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax2 = axes[1]
            if system.state_history:
                state_df = pd.DataFrame(system.state_history)
                state_df['datetime'] = state_df['timestamp'].apply(_ts_to_beijing_dt)
                state_df = state_df.sort_values('datetime')
                state_df['total_pnl'] = state_df['total_profit'] + state_df['unrealized_pnl']
                ax2.plot(state_df['datetime'], state_df['total_pnl'], label='累计盈亏', linewidth=2, color='blue')
                ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax2.fill_between(state_df['datetime'], 0, state_df['total_pnl'], where=(state_df['total_pnl'] >= 0), alpha=0.3, color='green', label='盈利区间')
                ax2.fill_between(state_df['datetime'], 0, state_df['total_pnl'], where=(state_df['total_pnl'] < 0), alpha=0.3, color='red', label='亏损区间')
            ax2.set_title('累计盈亏趋势', fontsize=14, fontweight='bold')
            ax2.set_ylabel('累计盈亏 (USDT)', fontsize=11)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
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
            output_path = os.path.join(output_dir, 'martin_bidirectional_results.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
            plt.close()
        try:
            print("\n生成图表...")
            plot_martin_bidirectional_results(df, system, results_dir)
            # 保存图表数据供仪表盘交互式 Plotly 使用
            if system.state_history:
                state_df = pd.DataFrame(system.state_history)
                state_df['datetime'] = state_df['timestamp'].apply(_ts_to_beijing_dt)
                state_df = state_df.sort_values('datetime')
                state_df['total_pnl'] = state_df['total_profit'] + state_df['unrealized_pnl']
                base_list = state_df['base_price'].tolist()
                base_list = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in base_list]
                chart_data = {
                    'datetime': state_df['datetime'].astype(str).tolist(),
                    'price': state_df['price'].tolist(),
                    'base_price': base_list,
                    'buy_level': state_df['buy_level'].tolist(),
                    'sell_level': state_df['sell_level'].tolist(),
                    'total_pnl': state_df['total_pnl'].tolist(),
                    'buy_count': state_df['buy_count'].tolist(),
                    'sell_count': state_df['sell_count'].tolist(),
                }
                buy_trades = [t for t in system.trade_history if t['action'] == 'open_buy']
                sell_trades = [t for t in system.trade_history if t['action'] == 'open_sell']
                close_trades = [t for t in system.trade_history if t['action'] == 'close_all']
                chart_data['trades_buy'] = [{'datetime': _ts_to_beijing_dt(t['timestamp']).strftime('%Y-%m-%d %H:%M:%S'), 'price': t['price']} for t in buy_trades]
                chart_data['trades_sell'] = [{'datetime': _ts_to_beijing_dt(t['timestamp']).strftime('%Y-%m-%d %H:%M:%S'), 'price': t['price']} for t in sell_trades]
                chart_data['trades_close'] = [{'datetime': _ts_to_beijing_dt(t['timestamp']).strftime('%Y-%m-%d %H:%M:%S'), 'price': t['price']} for t in close_trades]
                chart_path = os.path.join(results_dir, 'martin_bidirectional_chart_data.json')
                with open(chart_path, 'w', encoding='utf-8') as f:
                    json.dump(chart_data, f, ensure_ascii=False, indent=2, default=str)
                print(f"图表数据已保存到: {chart_path}")
        except Exception as e:
            print(f"绘图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    elif run_mode == 'realtime':
        # ===== 实时交易模式：可选仅信号 或 模拟盘下单 =====
        _fc = {}
        if args.config and os.path.isfile(args.config):
            try:
                with open(args.config, 'r', encoding='utf-8') as _f:
                    _fc = json.load(_f)
            except Exception:
                pass
        paper_trade = _fc.get('paper_trade', getattr(config, 'MARTIN_PAPER_TRADE', False))
        exchange_name = (data_config.get('exchange') or 'okx').strip().lower()
        symbol_cfg = data_config.get('symbol') or 'BTC-USDT-SWAP'
        proxies = data_config.get('proxies')
        interval_sec = max(1, int(poll_interval_seconds))

        api_client = None
        api_symbol = symbol_cfg
        if paper_trade:
            try:
                if exchange_name == 'okx':
                    from utils.api_clients import OKXAPI
                    api_client = OKXAPI(
                        api_key=config.OKX_CONFIG['api_key'],
                        api_secret=config.OKX_CONFIG['api_secret'],
                        passphrase=config.OKX_CONFIG['passphrase'],
                        sandbox=config.OKX_CONFIG.get('sandbox', True),
                    )
                    api_symbol = symbol_cfg  # 如 BTC-USDT-SWAP
                else:
                    from utils.api_clients import BinanceAPI
                    api_client = BinanceAPI(
                        api_key=config.BINANCE_CONFIG['api_key'],
                        api_secret=config.BINANCE_CONFIG['api_secret'],
                        testnet=config.BINANCE_CONFIG.get('testnet', True),
                    )
                    api_symbol = symbol_cfg.replace('-SWAP', '').replace('-', '')  # BTCUSDT
                print("\n模拟盘已开启：将根据策略信号在交易所测试环境下单（OKX 沙箱 / Binance 测试网）。")
            except Exception as e:
                print(f"\n模拟盘初始化失败: {e}，将仅运行信号不下单。")
                api_client = None
                paper_trade = False
        else:
            print("\n实时模式（仅信号）：不向交易所下单，按 Ctrl+C 停止。")

        system = BidirectionalMartinGrid(**strategy_config)
        exchange = data_config['exchange']
        symbol = data_config['symbol']
        # 实盘可视化：状态与历史写入 results/realtime_martin_live.json（供仪表盘读取）
        realtime_history = []  # 保留最近 N 条用于走势图
        REALTIME_HISTORY_MAX = 500
        live_file = os.path.join(config.RESULTS_DIR, 'realtime_martin_live.json')
        try:
            while True:
                rt = get_realtime_price(exchange, symbol, proxies=proxies)
                price, ts = rt.get('price', 0), rt.get('timestamp', 0)
                if price and price > 0 and ts:
                    prev_len = len(system.trade_history)
                    total_buy_size = sum(p['size'] for p in system.buy_positions)
                    total_sell_size = sum(p['size'] for p in system.sell_positions)
                    system.process_price(price, ts, atr=None)
                    status = system.get_status(price)
                    # 模拟盘：若有新交易则执行下单
                    if paper_trade and api_client and len(system.trade_history) > prev_len:
                        rec = system.trade_history[-1]
                        try:
                            if rec['action'] == 'open_buy':
                                side = 'buy' if exchange_name == 'okx' else 'BUY'
                                ret = api_client.place_order(api_symbol, side, rec['size'], order_type='market') if exchange_name == 'okx' else api_client.place_order(api_symbol, side, rec['size'], order_type='MARKET')
                                print(f"  [模拟盘] 开多 size={rec['size']} -> {ret.get('message', ret)}")
                            elif rec['action'] == 'open_sell':
                                side = 'sell' if exchange_name == 'okx' else 'SELL'
                                ret = api_client.place_order(api_symbol, side, rec['size'], order_type='market') if exchange_name == 'okx' else api_client.place_order(api_symbol, side, rec['size'], order_type='MARKET')
                                print(f"  [模拟盘] 开空 size={rec['size']} -> {ret.get('message', ret)}")
                            elif rec['action'] == 'close_all':
                                if total_buy_size > 0:
                                    side = 'sell' if exchange_name == 'okx' else 'SELL'
                                    if exchange_name == 'okx':
                                        ret = api_client.place_order(api_symbol, side, total_buy_size, order_type='market', reduce_only=True)
                                    else:
                                        ret = api_client.place_order(api_symbol, side, total_buy_size, order_type='MARKET', reduceOnly=True)
                                    print(f"  [模拟盘] 平多 size={total_buy_size} -> {ret.get('message', ret)}")
                                if total_sell_size > 0:
                                    side = 'buy' if exchange_name == 'okx' else 'BUY'
                                    if exchange_name == 'okx':
                                        ret = api_client.place_order(api_symbol, side, total_sell_size, order_type='market', reduce_only=True)
                                    else:
                                        ret = api_client.place_order(api_symbol, side, total_sell_size, order_type='MARKET', reduceOnly=True)
                                    print(f"  [模拟盘] 平空 size={total_sell_size} -> {ret.get('message', ret)}")
                        except Exception as ex:
                            print(f"  [模拟盘] 下单异常: {ex}")
                    print(f"[{datetime.datetime.now(BEIJING_TZ).strftime('%H:%M:%S')}] 价格={price:.2f} 买层={status['buy_level']} 卖层={status['sell_level']} 已实现盈亏={system.total_profit:.2f}")
                    # 写入实盘可视化数据（供仪表盘实盘页展示）
                    try:
                        now_iso = datetime.datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
                        realtime_history.append({
                            't': now_iso,
                            'price': round(price, 2),
                            'total_profit': round(system.total_profit, 4),
                            'buy_level': status['buy_level'],
                            'sell_level': status['sell_level'],
                        })
                        if len(realtime_history) > REALTIME_HISTORY_MAX:
                            realtime_history = realtime_history[-REALTIME_HISTORY_MAX:]
                        snapshot = {
                            'timestamp_iso': now_iso,
                            'price': round(price, 2),
                            'base_price': round(system.base_price, 2) if system.base_price is not None else None,
                            'buy_level': status['buy_level'],
                            'sell_level': status['sell_level'],
                            'total_profit': round(system.total_profit, 4),
                            'unrealized_pnl': round(status.get('unrealized_pnl', 0), 4),
                            'buy_positions_count': status['buy_positions_count'],
                            'sell_positions_count': status['sell_positions_count'],
                            'buy_avg_price': round(status['buy_avg_price'], 2) if status.get('buy_avg_price') is not None else None,
                            'sell_avg_price': round(status['sell_avg_price'], 2) if status.get('sell_avg_price') is not None else None,
                            'trade_count': status['trade_count'],
                        }
                        os.makedirs(config.RESULTS_DIR, exist_ok=True)
                        with open(live_file, 'w', encoding='utf-8') as f:
                            json.dump({'snapshot': snapshot, 'history': realtime_history}, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"  [实盘可视化] 写入失败: {e}")
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n用户中断，退出实时模式。")
        if system.trade_history or system.close_history:
            results_dir = config.RESULTS_DIR
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'martin_bidirectional_trades.json'), 'w', encoding='utf-8') as f:
                json.dump(system.trade_history, f, ensure_ascii=False, indent=2, default=str)
            with open(os.path.join(results_dir, 'martin_bidirectional_closes.json'), 'w', encoding='utf-8') as f:
                json.dump(system.close_history, f, ensure_ascii=False, indent=2, default=str)
            print(f"交易记录已保存到: {results_dir}")

