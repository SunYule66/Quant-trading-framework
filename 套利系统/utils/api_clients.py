"""
交易所API客户端 - 完整实现
支持OKX和Binance实时交易
"""

import logging
import time
import sys
import requests
from typing import Optional, Dict, List

# ===== OKX API 客户端 =====
class OKXAPI:
    """
    OKX API客户端
    需要安装: pip install okx
    """
    def __init__(self, api_key: str, api_secret: str, passphrase: str, sandbox: bool = False):
        """
        初始化OKX API
        
        Args:
            api_key: API密钥
            api_secret: API密钥
            passphrase: API密钥密码
            sandbox: 是否使用沙箱环境
        """
        try:
            import ccxt
            self.exchange = ccxt.okx({
                'apiKey': api_key,
                'secret': api_secret,
                'password': passphrase,
                'sandbox': sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # 默认使用永续合约
                    'defaultMarket': 'swap',  # 明确指定市场类型
                }
            })
            # 禁用自动重新加载markets，避免加载SPOT市场
            # markets会在实际调用时按需加载
            if hasattr(self.exchange, 'reload_markets'):
                self.exchange.reload_markets = False
            self.sandbox = sandbox
            # 不预加载市场，避免加载SPOT市场导致错误
            # 市场会在实际调用时按需加载，并且会使用defaultType='swap'
            logging.info(f"OKX API客户端初始化成功 (sandbox={sandbox}, defaultType=swap)")
        except ImportError as e:
            error_msg = (
                "请安装ccxt库: pip install ccxt\n"
                "或者运行: pip install -r 套利系统/requirements.txt\n"
                f"当前Python路径: {sys.executable}\n"
                f"原始错误: {e}"
            )
            raise ImportError(error_msg)
        except Exception as e:
            logging.error(f"OKX API初始化失败: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        获取最新价格
        
        Args:
            symbol: 交易对，如 'BTC-USDT-SWAP'
            
        Returns:
            dict: 价格信息 {'last': price, 'bid': bid_price, 'ask': ask_price}
        """
        # 转换symbol格式
        if '-SWAP' in symbol:
            ccxt_symbol = symbol.replace('-SWAP', '').replace('-', '/') + ':USDT'
        else:
            ccxt_symbol = symbol.replace('-', '/')
        
        okx_symbol = symbol
        
        # 使用直接REST API调用，添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 构建API URL
                base_url = 'https://www.okx.com' if not self.sandbox else 'https://www.okx.com'
                url = f'{base_url}/api/v5/market/ticker'
                params = {'instId': okx_symbol}
                
                # 直接调用OKX REST API，添加重试和超时设置
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=10,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('code') == '0' and data.get('data'):
                    ticker_data = data['data'][0]
                    return {
                        'last': float(ticker_data.get('last', 0)),
                        'bid': float(ticker_data.get('bidPx', 0)) if ticker_data.get('bidPx') else float(ticker_data.get('last', 0)),
                        'ask': float(ticker_data.get('askPx', 0)) if ticker_data.get('askPx') else float(ticker_data.get('last', 0)),
                        'timestamp': int(ticker_data.get('ts', time.time() * 1000)) // 1000
                    }
                else:
                    error_msg = data.get('msg', '未知错误')
                    logging.error(f"OKX获取价格失败: {error_msg}, symbol={okx_symbol}")
                    break  # API返回错误，不需要重试
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    logging.warning(f"OKX网络连接失败，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"OKX网络连接失败，已重试{max_retries}次: {e}")
            except requests.RequestException as e:
                logging.warning(f"OKX API请求失败: {e}")
                break  # 其他请求错误，不需要重试
            except Exception as e:
                logging.warning(f"OKX API调用异常: {e}")
                break  # 其他异常，不需要重试
        
        # 如果所有方法都失败，返回默认值
        logging.error(f"OKX获取价格失败: 所有方法都失败, symbol={symbol}")
        return {'last': 0, 'bid': 0, 'ask': 0, 'timestamp': int(time.time())}
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        获取资金费率
        
        Args:
            symbol: 交易对
            
        Returns:
            dict: 资金费率信息 {'fundingRate': rate, 'nextFundingTime': timestamp}
        """
        okx_symbol = symbol
        
        # 使用直接REST API调用，添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 构建API URL
                base_url = 'https://www.okx.com' if not self.sandbox else 'https://www.okx.com'
                url = f'{base_url}/api/v5/public/funding-rate'
                params = {'instId': okx_symbol}
                
                # 直接调用OKX REST API
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=10,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('code') == '0' and data.get('data'):
                    funding_data = data['data'][0]
                    return {
                        'fundingRate': float(funding_data.get('fundingRate', 0)),
                        'nextFundingTime': int(funding_data.get('nextFundingTime', 0)) // 1000 if funding_data.get('nextFundingTime') else 0
                    }
                else:
                    error_msg = data.get('msg', '未知错误')
                    logging.error(f"OKX获取资金费率失败: {error_msg}, symbol={okx_symbol}")
                    break  # API返回错误，不需要重试
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logging.warning(f"OKX网络连接失败，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"OKX网络连接失败，已重试{max_retries}次: {e}")
            except requests.RequestException as e:
                logging.warning(f"OKX API请求失败: {e}")
                break
            except Exception as e:
                logging.warning(f"OKX API调用异常: {e}")
                break
        
        # 如果所有重试都失败，返回默认值
        logging.error(f"OKX获取资金费率失败: 所有方法都失败, symbol={symbol}")
        return {'fundingRate': 0, 'nextFundingTime': 0}
    
    def get_klines(self, symbol: str, interval: str = '1H', limit: int = 100) -> List:
        """
        获取K线数据
        
        Args:
            symbol: 交易对
            interval: 时间间隔，如 '1H', '4H', '1D'
            limit: 数量
            
        Returns:
            list: K线数据列表，每个元素包含 [timestamp, open, high, low, close, volume]
        """
        try:
            # 转换symbol格式
            if '-SWAP' in symbol:
                ccxt_symbol = symbol.replace('-SWAP', '').replace('-', '/') + ':USDT'
            else:
                ccxt_symbol = symbol.replace('-', '/')
            
            # ccxt使用标准时间间隔格式
            ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, interval, limit=limit)
            klines = []
            for candle in ohlcv:
                klines.append({
                    'timestamp': int(candle[0]) // 1000,
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            return klines
        except Exception as e:
            logging.error(f"OKX获取K线异常: symbol={symbol}, error={e}")
            return []
    
    def place_order(self, symbol: str, side: str, size: float, 
                   order_type: str = 'market', price: Optional[float] = None,
                   reduce_only: bool = False) -> Dict:
        """
        下单
        
        Args:
            symbol: 交易对
            side: 方向 'buy' 或 'sell'
            size: 数量（合约张数）
            order_type: 订单类型 'market' 或 'limit'
            price: 限价单价格（限价单必填）
            reduce_only: 是否只减仓
            
        Returns:
            dict: 订单信息 {'orderId': id, 'status': status, 'filled': filled_size}
        """
        try:
            # 转换symbol格式
            if '-SWAP' in symbol:
                ccxt_symbol = symbol.replace('-SWAP', '').replace('-', '/') + ':USDT'
            else:
                ccxt_symbol = symbol.replace('-', '/')
            
            # 构建订单参数
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            if order_type == 'market':
                order = self.exchange.create_market_order(ccxt_symbol, side, size, None, params)
            else:
                if price is None:
                    raise ValueError("限价单必须提供价格")
                order = self.exchange.create_limit_order(ccxt_symbol, side, size, price, params)
            
            return {
                'orderId': str(order.get('id', '')),
                'status': order.get('status', ''),
                'filled': float(order.get('filled', 0)),
                'message': 'success'
            }
        except Exception as e:
            logging.error(f"OKX下单异常: symbol={symbol}, error={e}")
            return {
                'orderId': '',
                'status': 'failed',
                'filled': 0,
                'message': str(e)
            }
    
    def get_position(self, symbol: str) -> Dict:
        """
        获取持仓信息
        
        Args:
            symbol: 交易对
            
        Returns:
            dict: 持仓信息 {'size': size, 'side': side, 'avgPrice': price}
        """
        try:
            # 转换symbol格式
            if '-SWAP' in symbol:
                ccxt_symbol = symbol.replace('-SWAP', '').replace('-', '/') + ':USDT'
            else:
                ccxt_symbol = symbol.replace('-', '/')
            
            positions = self.exchange.fetch_positions([ccxt_symbol])
            if positions:
                for pos in positions:
                    if pos.get('contracts', 0) != 0:
                        return {
                            'size': abs(float(pos.get('contracts', 0))),
                            'side': 'long' if float(pos.get('contracts', 0)) > 0 else 'short',
                            'avgPrice': float(pos.get('entryPrice', 0)),
                            'unrealizedPnl': float(pos.get('unrealizedPnl', 0))
                        }
            return {'size': 0, 'side': 'none', 'avgPrice': 0, 'unrealizedPnl': 0}
        except Exception as e:
            logging.error(f"OKX获取持仓异常: symbol={symbol}, error={e}")
            return {'size': 0, 'side': 'none', 'avgPrice': 0, 'unrealizedPnl': 0}


# ===== Binance API 客户端 =====
class BinanceAPI:
    """
    Binance API客户端
    需要安装: pip install python-binance
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        初始化Binance API
        
        Args:
            api_key: API密钥
            api_secret: API密钥
            testnet: 是否使用测试网
        """
        try:
            from binance.client import Client
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
            self.testnet = testnet
            logging.info(f"Binance API客户端初始化成功 (testnet={testnet})")
        except ImportError as e:
            error_msg = (
                "请安装python-binance库: pip install python-binance\n"
                "或者运行: pip install -r 套利系统/requirements.txt\n"
                f"当前Python路径: {sys.executable}\n"
                f"原始错误: {e}"
            )
            raise ImportError(error_msg)
        except Exception as e:
            logging.error(f"Binance API初始化失败: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        获取最新价格
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            
        Returns:
            dict: 价格信息 {'lastPrice': price, 'bidPrice': bid, 'askPrice': ask}
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return {
                'lastPrice': float(ticker.get('price', 0)),
                'bidPrice': 0,  # Binance ticker不直接提供bid/ask
                'askPrice': 0,
                'timestamp': int(time.time())
            }
        except Exception as e:
            logging.error(f"Binance获取价格异常: {e}")
            return {'lastPrice': 0, 'bidPrice': 0, 'askPrice': 0, 'timestamp': int(time.time())}
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        获取资金费率
        
        Args:
            symbol: 交易对
            
        Returns:
            dict: 资金费率信息 {'lastFundingRate': rate, 'nextFundingTime': timestamp}
        """
        try:
            funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if funding_rates:
                data = funding_rates[0]
                return {
                    'lastFundingRate': float(data.get('lastFundingRate', 0)),
                    'nextFundingTime': int(data.get('fundingTime', 0)) // 1000
                }
            else:
                logging.error(f"Binance获取资金费率失败: 无数据")
                return {'lastFundingRate': 0, 'nextFundingTime': 0}
        except Exception as e:
            logging.error(f"Binance获取资金费率异常: {e}")
            return {'lastFundingRate': 0, 'nextFundingTime': 0}
    
    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List:
        """
        获取K线数据
        
        Args:
            symbol: 交易对
            interval: 时间间隔，如 '1m', '5m', '1h', '4h', '1d'
            limit: 数量
            
        Returns:
            list: K线数据列表，每个元素包含 [timestamp, open, high, low, close, volume]
        """
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            result = []
            for kline in klines:
                result.append({
                    'timestamp': int(kline[0]) // 1000,
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            return result
        except Exception as e:
            logging.error(f"Binance获取K线异常: {e}")
            return []
    
    def place_order(self, symbol: str, side: str, quantity: float,
                   order_type: str = 'MARKET', price: Optional[float] = None,
                   reduceOnly: bool = False) -> Dict:
        """
        下单
        
        Args:
            symbol: 交易对
            side: 方向 'BUY' 或 'SELL'
            quantity: 数量
            order_type: 订单类型 'MARKET' 或 'LIMIT'
            price: 限价单价格（限价单必填）
            reduceOnly: 是否只减仓
            
        Returns:
            dict: 订单信息 {'orderId': id, 'status': status, 'filled': filled_qty}
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
            }
            
            if order_type == 'LIMIT' and price:
                params['timeInForce'] = 'GTC'
                params['price'] = price
            
            if reduceOnly:
                params['reduceOnly'] = True
            
            response = self.client.futures_create_order(**params)
            
            return {
                'orderId': str(response.get('orderId', '')),
                'status': response.get('status', ''),
                'filled': float(response.get('executedQty', 0)),
                'message': 'success'
            }
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Binance下单异常: {error_msg}")
            return {
                'orderId': '',
                'status': 'FAILED',
                'filled': 0,
                'message': error_msg
            }
    
    def get_position(self, symbol: str) -> Dict:
        """
        获取持仓信息
        
        Args:
            symbol: 交易对
            
        Returns:
            dict: 持仓信息 {'size': size, 'side': side, 'avgPrice': price}
        """
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if positions:
                pos = positions[0]
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:
                    return {
                        'size': abs(position_amt),
                        'side': 'long' if position_amt > 0 else 'short',
                        'avgPrice': float(pos.get('entryPrice', 0)),
                        'unrealizedPnl': float(pos.get('unRealizedProfit', 0))
                    }
            return {'size': 0, 'side': 'none', 'avgPrice': 0, 'unrealizedPnl': 0}
        except Exception as e:
            logging.error(f"Binance获取持仓异常: {e}")
            return {'size': 0, 'side': 'none', 'avgPrice': 0, 'unrealizedPnl': 0}


# ===== 使用示例 =====

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：初始化API客户端
    try:
        # OKX API（需要从config.py或环境变量读取）
        okx_api = OKXAPI(
            api_key='your_okx_api_key',
            api_secret='your_okx_api_secret',
            passphrase='your_okx_passphrase',
            sandbox=True  # 测试环境
        )
        
        # 测试获取价格
        ticker = okx_api.get_ticker('BTC-USDT-SWAP')
        print(f"OKX BTC价格: {ticker['last']}")
        
        # 测试获取资金费率
        funding = okx_api.get_funding_rate('BTC-USDT-SWAP')
        print(f"OKX 资金费率: {funding['fundingRate']}")
        
    except Exception as e:
        print(f"OKX API测试失败: {e}")
    
    try:
        # Binance API（需要从config.py或环境变量读取）
        binance_api = BinanceAPI(
            api_key='your_binance_api_key',
            api_secret='your_binance_api_secret',
            testnet=True  # 测试环境
        )
        
        # 测试获取价格
        ticker = binance_api.get_ticker('BTCUSDT')
        print(f"Binance BTC价格: {ticker['lastPrice']}")
        
        # 测试获取资金费率
        funding = binance_api.get_funding_rate('BTCUSDT')
        print(f"Binance 资金费率: {funding['lastFundingRate']}")
        
    except Exception as e:
        print(f"Binance API测试失败: {e}")
    
    print("\nAPI客户端已就绪，可以在realtime_trading.py中使用")

