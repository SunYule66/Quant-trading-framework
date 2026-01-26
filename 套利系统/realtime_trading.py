"""
实时套利交易系统
将回测逻辑转换为实时交易系统
"""

import time
import logging
import json
from datetime import datetime
from logic import ArbitrageSystem
from api_clients import OKXAPI, BinanceAPI
import pandas as pd
import config

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_CONFIG['level']),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_CONFIG['file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class RealtimeArbitrageTrader:
    def __init__(self, system: ArbitrageSystem, okx_api, binance_api):
        """
        初始化实时交易系统
        
        Args:
            system: 套利系统实例
            okx_api: OKX API客户端
            binance_api: Binance API客户端
        """
        self.system = system
        self.okx_api = okx_api
        self.binance_api = binance_api
        self.running = False
        self.total_capital = config.TRADING_CONFIG['total_capital']
        self.leverage = config.TRADING_CONFIG['leverage']
        self.symbol_okx = 'BTC-USDT-SWAP'
        self.symbol_binance = 'BTCUSDT'
        
    def get_current_data(self):
        """
        获取当前时刻的价格和资金费率数据
        
        Returns:
            dict: 包含price_a, price_b, funding_a, funding_b, timestamp的字典
        """
        try:
            # 获取OKX价格
            okx_ticker = self.okx_api.get_ticker(self.symbol_okx)
            okx_price = float(okx_ticker.get('last', 0))
            
            if okx_price == 0:
                logging.error("OKX价格获取失败")
                return None
            
            # 获取Binance价格
            binance_ticker = self.binance_api.get_ticker(self.symbol_binance)
            binance_price = float(binance_ticker.get('lastPrice', 0))
            
            if binance_price == 0:
                logging.error("Binance价格获取失败")
                return None
            
            # 获取OKX资金费率
            okx_funding = self.okx_api.get_funding_rate(self.symbol_okx)
            okx_funding_rate = float(okx_funding.get('fundingRate', 0))
            
            # 获取Binance资金费率
            binance_funding = self.binance_api.get_funding_rate(self.symbol_binance)
            binance_funding_rate = float(binance_funding.get('lastFundingRate', 0))
            
            current_ts = int(time.time())
            
            return {
                'price_a': okx_price,
                'price_b': binance_price,
                'funding_a': okx_funding_rate,
                'funding_b': binance_funding_rate,
                'timestamp': current_ts
            }
        except Exception as e:
            logging.error(f"获取数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_history_data(self, hours=5):
        """
        获取历史N小时的数据（用于计算历史平均值）
        
        Args:
            hours: 历史小时数
            
        Returns:
            pd.DataFrame: 历史数据，包含price_a, price_b, funding_a, funding_b，时间戳作为索引
        """
        try:
            # 获取OKX历史K线数据
            okx_klines = self.okx_api.get_klines(self.symbol_okx, '1H', hours)
            # 获取Binance历史K线数据
            binance_klines = self.binance_api.get_klines(self.symbol_binance, '1h', hours)
            
            if not okx_klines or not binance_klines:
                logging.warning("历史K线数据为空")
                return pd.DataFrame()
            
            # 创建时间戳到数据的映射
            okx_dict = {k['timestamp']: k for k in okx_klines}
            binance_dict = {k['timestamp']: k for k in binance_klines}
            
            # 获取所有时间戳的交集
            common_timestamps = sorted(set(okx_dict.keys()) & set(binance_dict.keys()))
            
            if not common_timestamps:
                logging.warning("没有共同的时间戳")
                return pd.DataFrame()
            
            # 构建历史数据列表
            history_data = []
            for ts in common_timestamps:
                okx_data = okx_dict[ts]
                binance_data = binance_dict[ts]
                
                # 获取该时间点的资金费率（需要实时获取，这里简化处理）
                # 实际应该获取历史资金费率，但API可能不支持，这里使用当前资金费率作为近似
                okx_funding = self.okx_api.get_funding_rate(self.symbol_okx)['fundingRate']
                binance_funding = self.binance_api.get_funding_rate(self.symbol_binance)['lastFundingRate']
                
                history_data.append({
                    'timestamp': ts,
                    'price_a': okx_data['close'],  # 使用收盘价
                    'price_b': binance_data['close'],
                    'funding_a': okx_funding,
                    'funding_b': binance_funding
                })
            
            # 转换为DataFrame，时间戳作为索引
            df = pd.DataFrame(history_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            logging.error(f"获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def execute_open_position(self, position_info):
        """
        执行开仓操作
        
        Args:
            position_info: 开仓信息字典
        """
        try:
            price_spread = position_info['开仓差价']
            
            # 计算持仓数量
            okx_size = self.calculate_position_size(position_info['开仓价格a'], 0.5)
            binance_size = self.calculate_position_size(position_info['开仓价格b'], 0.5)
            
            if price_spread >= 0:
                # 价差>0: 做空OKX，做多Binance
                logging.info(f"开仓: 做空OKX @ {position_info['开仓价格a']}, 做多Binance @ {position_info['开仓价格b']}")
                
                # OKX做空
                okx_order = self.okx_api.place_order(
                    symbol=self.symbol_okx,
                    side='sell',
                    size=okx_size,
                    order_type='market'
                )
                
                # Binance做多
                binance_order = self.binance_api.place_order(
                    symbol=self.symbol_binance,
                    side='BUY',
                    quantity=binance_size,
                    order_type='MARKET'
                )
                
                # 记录实际成交数量
                position_info['持仓数量_a'] = okx_order.get('filled', okx_size)
                position_info['持仓数量_b'] = binance_order.get('filled', binance_size)
                
                logging.info(f"OKX订单: {okx_order}")
                logging.info(f"Binance订单: {binance_order}")
                
            else:
                # 价差<0: 做多OKX，做空Binance
                logging.info(f"开仓: 做多OKX @ {position_info['开仓价格a']}, 做空Binance @ {position_info['开仓价格b']}")
                
                # OKX做多
                okx_order = self.okx_api.place_order(
                    symbol=self.symbol_okx,
                    side='buy',
                    size=okx_size,
                    order_type='market'
                )
                
                # Binance做空
                binance_order = self.binance_api.place_order(
                    symbol=self.symbol_binance,
                    side='SELL',
                    quantity=binance_size,
                    order_type='MARKET'
                )
                
                # 记录实际成交数量
                position_info['持仓数量_a'] = okx_order.get('filled', okx_size)
                position_info['持仓数量_b'] = binance_order.get('filled', binance_size)
                
                logging.info(f"OKX订单: {okx_order}")
                logging.info(f"Binance订单: {binance_order}")
            
            # 检查订单是否成功
            if okx_order.get('status') == 'failed' or binance_order.get('status') == 'FAILED':
                logging.error(f"开仓失败: OKX={okx_order.get('message')}, Binance={binance_order.get('message')}")
                # 如果一边失败，需要撤销另一边（这里简化处理，实际应该实现撤销逻辑）
                
        except Exception as e:
            logging.error(f"开仓执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def execute_close_position(self, position_info):
        """
        执行平仓操作
        
        Args:
            position_info: 持仓信息字典
        """
        try:
            price_spread = position_info['开仓差价']
            
            # 获取持仓数量（如果之前没有记录，从API获取）
            okx_size = position_info.get('持仓数量_a', 0)
            binance_size = position_info.get('持仓数量_b', 0)
            
            # 如果持仓数量未记录，尝试从API获取
            if okx_size == 0 or binance_size == 0:
                okx_pos = self.okx_api.get_position(self.symbol_okx)
                binance_pos = self.binance_api.get_position(self.symbol_binance)
                okx_size = okx_size if okx_size > 0 else abs(okx_pos.get('size', 0))
                binance_size = binance_size if binance_size > 0 else abs(binance_pos.get('size', 0))
            
            if okx_size == 0 or binance_size == 0:
                logging.warning(f"持仓数量为0，无法平仓: OKX={okx_size}, Binance={binance_size}")
                return
            
            if price_spread >= 0:
                # 平仓: 平空OKX，平多Binance
                reason = position_info.get('平仓信息', {}).get('平仓原因', '未知原因')
                logging.info(f"平仓: 平空OKX, 平多Binance, 原因: {reason}")
                
                # OKX平空（买入平仓）
                okx_order = self.okx_api.place_order(
                    symbol=self.symbol_okx,
                    side='buy',
                    size=okx_size,
                    order_type='market',
                    reduce_only=True
                )
                
                # Binance平多（卖出平仓）
                binance_order = self.binance_api.place_order(
                    symbol=self.symbol_binance,
                    side='SELL',
                    quantity=binance_size,
                    order_type='MARKET',
                    reduceOnly=True
                )
                
            else:
                # 平仓: 平多OKX，平空Binance
                reason = position_info.get('平仓信息', {}).get('平仓原因', '未知原因')
                logging.info(f"平仓: 平多OKX, 平空Binance, 原因: {reason}")
                
                # OKX平多（卖出平仓）
                okx_order = self.okx_api.place_order(
                    symbol=self.symbol_okx,
                    side='sell',
                    size=okx_size,
                    order_type='market',
                    reduce_only=True
                )
                
                # Binance平空（买入平仓）
                binance_order = self.binance_api.place_order(
                    symbol=self.symbol_binance,
                    side='BUY',
                    quantity=binance_size,
                    order_type='MARKET',
                    reduceOnly=True
                )
            
            logging.info(f"平仓订单: OKX={okx_order}, Binance={binance_order}")
            
            # 检查订单是否成功
            if okx_order.get('status') == 'failed' or binance_order.get('status') == 'FAILED':
                logging.error(f"平仓失败: OKX={okx_order.get('message')}, Binance={binance_order.get('message')}")
            
        except Exception as e:
            logging.error(f"平仓执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_position_size(self, price, capital_ratio):
        """
        计算持仓数量
        
        Args:
            price: 价格
            capital_ratio: 资金比例（0.5表示50%）
            
        Returns:
            float: 持仓数量（合约张数或数量）
        """
        # 从配置读取总资金和杠杆
        position_value = self.total_capital * capital_ratio
        # 考虑杠杆
        position_size = (position_value * self.leverage) / price
        
        # OKX使用合约张数，Binance使用数量
        # 这里返回的是基础数量，实际使用时需要根据交易所要求调整
        return round(position_size, 4)  # 保留4位小数
    
    def check_and_trade(self):
        """
        检查市场条件并执行交易
        """
        try:
            # 获取当前数据
            current_data = self.get_current_data()
            if current_data is None:
                logging.warning("获取当前数据失败，跳过本次检查")
                return
            
            # 获取历史数据
            history_df = self.get_history_data(self.system.N)
            
            # 构建包含历史数据和当前数据的DataFrame
            # 将当前数据添加到历史数据中
            current_ts = current_data['timestamp']
            current_row = {
                'price_a': current_data['price_a'],
                'price_b': current_data['price_b'],
                'funding_a': current_data['funding_a'],
                'funding_b': current_data['funding_b']
            }
            
            # 创建包含历史+当前的DataFrame
            if not history_df.empty:
                df = history_df.copy()
                df.loc[current_ts] = current_row
            else:
                # 如果历史数据为空，只使用当前数据
                df = pd.DataFrame([current_row], index=[current_ts])
            
            df.sort_index(inplace=True)
            
            # 找到当前数据在DataFrame中的索引
            current_idx = len(df) - 1
            
            # 检查平仓条件
            if self.system.has_open_positions():
                closed_any = self.system.check_close(df, current_idx)
                if closed_any:
                    # 找到刚平仓的持仓并执行平仓
                    for pos in self.system.positions:
                        if pos.get('平仓') and pos.get('平仓信息') is not None:
                            # 检查是否已经执行过平仓（通过检查是否有订单记录）
                            if '平仓订单' not in pos:
                                self.execute_close_position(pos)
                                pos['平仓订单'] = True  # 标记已执行
                                logging.info(f"已执行平仓: {pos.get('平仓信息', {}).get('平仓原因', '未知')}")
            
            # 检查开仓条件（仅在无持仓时）
            if not self.system.has_open_positions():
                position = self.system.check_open(df, current_idx)
                if position:
                    logging.info(f"满足开仓条件: {position['触发模式']} - {position['触发条件']}")
                    self.execute_open_position(position)
                    
        except Exception as e:
            logging.error(f"检查交易条件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self, interval=60):
        """
        运行实时交易循环
        
        Args:
            interval: 检查间隔（秒），默认60秒
        """
        self.running = True
        logging.info("实时套利交易系统启动")
        
        last_save_time = time.time()
        save_interval = 300  # 每5分钟保存一次
        
        while self.running:
            try:
                self.check_and_trade()
                
                # 定期保存持仓记录
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    self.save_positions()
                    last_save_time = current_time
                
                time.sleep(interval)
            except KeyboardInterrupt:
                logging.info("收到停止信号，正在关闭...")
                self.running = False
                break
            except Exception as e:
                logging.error(f"交易循环错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(interval)
        
        logging.info("实时套利交易系统已停止")
        self.save_positions()  # 最后保存一次
    
    def stop(self):
        """停止交易系统"""
        self.running = False
        # 保存持仓记录
        self.save_positions()
    
    def save_positions(self):
        """保存持仓记录到JSON文件"""
        try:
            def convert(obj):
                """转换numpy类型为Python原生类型"""
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj
            
            positions = convert(self.system.positions)
            with open('套利系统/arbitrage_positions.json', 'w', encoding='utf-8') as f:
                json.dump(positions, f, ensure_ascii=False, indent=2)
            logging.info("持仓记录已保存到 arbitrage_positions.json")
        except Exception as e:
            logging.error(f"保存持仓记录失败: {e}")


# 使用示例
if __name__ == "__main__":
    try:
        # 初始化套利系统（使用config中的参数）
        system = ArbitrageSystem(**config.ARBITRAGE_CONFIG)
        
        # 初始化API客户端
        logging.info("正在初始化API客户端...")
        okx_api = OKXAPI(**config.OKX_CONFIG)
        binance_api = BinanceAPI(**config.BINANCE_CONFIG)
        
        # 创建实时交易器
        trader = RealtimeArbitrageTrader(system, okx_api, binance_api)
        
        # 运行交易系统
        logging.info("实时套利交易系统启动")
        logging.info(f"检查间隔: {config.TRADING_CONFIG['check_interval']}秒")
        logging.info(f"总资金: {config.TRADING_CONFIG['total_capital']} USDT")
        logging.info(f"杠杆: {config.TRADING_CONFIG['leverage']}x")
        
        trader.run(interval=config.TRADING_CONFIG['check_interval'])
        
    except KeyboardInterrupt:
        logging.info("收到停止信号，正在关闭...")
        if 'trader' in locals():
            trader.stop()
    except Exception as e:
        logging.error(f"系统启动失败: {e}")
        import traceback
        traceback.print_exc()

