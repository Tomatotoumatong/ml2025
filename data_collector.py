import asyncio
import aiohttp
import websockets
import json
import time
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from enum import Enum
import aiohttp
# 导入基础设施模块
from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils, RetryDecorator, async_retry
from database_manager import DatabaseManager


class DataSource(Enum):
    """数据源枚举"""
    BINANCE = "binance"
    OKEX = "okex"
    HUOBI = "huobi"
    BYBIT = "bybit"


class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MarketDataCollector:
    """实时市场数据采集器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("DATA_COLLECTOR")
        self.db_manager = DatabaseManager(config_path)
        
        # 连接状态管理
        self.connection_states: Dict[str, ConnectionState] = {}
        self.websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[str]] = {}
        
        # 数据回调
        self.data_callbacks: List[Callable] = []
        
        # 重连配置
        self.max_reconnect_attempts = self.config.get("collector.max_reconnect_attempts", 10)
        self.reconnect_delay = self.config.get("collector.reconnect_delay", 5)
        self.heartbeat_interval = self.config.get("collector.heartbeat_interval", 30)
        # 代理配置
        self.proxy_config = {
            'http': self.config.get("network.proxy_http", "http://127.0.0.1:7890"),
            'https': self.config.get("network.proxy_https", "http://127.0.0.1:7890"),
            'enabled': self.config.get("network.proxy_enabled", True)
        }
        # 支持的交易所配置
        self.exchanges = {
            DataSource.BINANCE: {
                "ws_url": "wss://stream.binance.com:9443/ws/",
                "rest_url": "https://api.binance.com/api/v3/",
                "rate_limit": 1200  # 每分钟请求数
            },
            DataSource.OKEX: {
                "ws_url": "wss://ws.okex.com:8443/ws/v5/public",
                "rest_url": "https://www.okex.com/api/v5/",
                "rate_limit": 600
            }
        }
        
        # 启动任务
        self.running_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    async def start(self, symbols: List[str], exchanges: List[DataSource] = None):
        """启动数据采集"""
        if self.is_running:
            self.logger.warning("数据采集器已在运行")
            return
        
        self.is_running = True
        exchanges = exchanges or [DataSource.BINANCE]
        
        self.logger.info(f"启动数据采集 - 交易对: {symbols}, 交易所: {[e.value for e in exchanges]}")
        
        # 为每个交易所启动WebSocket连接
        for exchange in exchanges:
            task = asyncio.create_task(
                self._start_websocket_connection(exchange, symbols)
            )
            self.running_tasks.append(task)
        
        # 启动REST轮询备份
        rest_task = asyncio.create_task(
            self._start_rest_polling(symbols, exchanges)
        )
        self.running_tasks.append(rest_task)
        
        # 启动数据质量监控
        monitor_task = asyncio.create_task(self._monitor_data_quality())
        self.running_tasks.append(monitor_task)
    
    async def _start_websocket_connection(self, exchange: DataSource, symbols: List[str]):
        """启动WebSocket连接"""
        exchange_name = exchange.value
        config = self.exchanges[exchange]
        reconnect_count = 0
        
        while self.is_running and reconnect_count < self.max_reconnect_attempts:
            try:
                self.connection_states[exchange_name] = ConnectionState.CONNECTING
                self.logger.info(f"连接到 {exchange_name} WebSocket...")
                
                # 构建订阅URL
                ws_url = self._build_websocket_url(exchange, symbols)
                
                proxy_connector = aiohttp.TCPConnector()
                
                async with websockets.connect(
                    ws_url,
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websockets[exchange_name] = websocket
                    self.connection_states[exchange_name] = ConnectionState.CONNECTED
                    self.subscriptions[exchange_name] = symbols
                    reconnect_count = 0
                    
                    self.logger.info(f"{exchange_name} WebSocket连接成功")
                    
                    # 发送订阅消息
                    await self._send_subscription(websocket, exchange, symbols)
                    
                    # 监听数据
                    async for message in websocket:
                        try:
                            await self._process_websocket_message(exchange, message)
                        except Exception as e:
                            self.logger.error(f"处理WebSocket消息失败: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"{exchange_name} WebSocket连接关闭")
            except Exception as e:
                self.logger.error(f"{exchange_name} WebSocket连接错误: {e}")
            
            if self.is_running:
                self.connection_states[exchange_name] = ConnectionState.RECONNECTING
                reconnect_count += 1
                wait_time = min(self.reconnect_delay * reconnect_count, 60)
                self.logger.info(f"{exchange_name} 等待 {wait_time}s 后重连 (尝试 {reconnect_count}/{self.max_reconnect_attempts})")
                await asyncio.sleep(wait_time)
        
        if reconnect_count >= self.max_reconnect_attempts:
            self.connection_states[exchange_name] = ConnectionState.ERROR
            self.logger.error(f"{exchange_name} 达到最大重连次数，停止重连")
    
    def _build_websocket_url(self, exchange: DataSource, symbols: List[str]) -> str:
        """构建WebSocket订阅URL"""
        base_url = self.exchanges[exchange]["ws_url"]
        
        if exchange == DataSource.BINANCE:
            # Binance格式: /ws/btcusdt@ticker/ethusdt@ticker
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                streams.extend([
                    f"{symbol_lower}@ticker",      # 24hr价格统计
                    f"{symbol_lower}@depth20",     # 深度数据
                    f"{symbol_lower}@trade"        # 实时交易
                ])
            return f"{base_url}{'/'.join(streams)}"
        
        elif exchange == DataSource.OKEX:
            return base_url
        
        return base_url
    
    async def _send_subscription(self, websocket, exchange: DataSource, symbols: List[str]):
        """发送订阅消息"""
        if exchange == DataSource.OKEX:
            # OKEx需要发送订阅消息
            subscribe_msg = {
                "op": "subscribe",
                "args": []
            }
            
            for symbol in symbols:
                symbol_formatted = symbol.upper().replace('USDT', '-USDT')
                subscribe_msg["args"].extend([
                    {"channel": "tickers", "instId": symbol_formatted},
                    {"channel": "books5", "instId": symbol_formatted},
                    {"channel": "trades", "instId": symbol_formatted}
                ])
            
            await websocket.send(json.dumps(subscribe_msg))
    
    async def _process_websocket_message(self, exchange: DataSource, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            if exchange == DataSource.BINANCE:
                await self._process_binance_message(data)
            elif exchange == DataSource.OKEX:
                await self._process_okex_message(data)
                
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析失败: {message}")
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
    
    async def _process_binance_message(self, data: Dict[str, Any]):
        """处理Binance消息"""
        stream = data.get('stream', '')
        event_data = data.get('data', {})
        
        if '@ticker' in stream:
            # 24小时价格统计
            await self._handle_ticker_data(DataSource.BINANCE, event_data)
        elif '@depth' in stream:
            # 深度数据
            await self._handle_depth_data(DataSource.BINANCE, event_data)
        elif '@trade' in stream:
            # 实时交易
            await self._handle_trade_data(DataSource.BINANCE, event_data)
    
    async def _process_okex_message(self, data: Dict[str, Any]):
        """处理OKEx消息"""
        if 'data' not in data:
            return
        
        arg = data.get('arg', {})
        channel = arg.get('channel')
        
        for item in data['data']:
            if channel == 'tickers':
                await self._handle_ticker_data(DataSource.OKEX, item)
            elif channel == 'books5':
                await self._handle_depth_data(DataSource.OKEX, item)
            elif channel == 'trades':
                await self._handle_trade_data(DataSource.OKEX, item)
    
    async def _handle_ticker_data(self, exchange: DataSource, data: Dict[str, Any]):
        """处理价格统计数据"""
        try:
            if exchange == DataSource.BINANCE:
                symbol = data.get('s', '').upper()
                timestamp = TimeUtils.now_timestamp()
                
                market_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(data.get('o', 0)),
                    'high': float(data.get('h', 0)),
                    'low': float(data.get('l', 0)),
                    'close': float(data.get('c', 0)),
                    'volume': float(data.get('v', 0)),
                    'exchange': exchange.value
                }
                
            elif exchange == DataSource.OKEX:
                symbol = data.get('instId', '').replace('-', '').upper()
                timestamp = int(data.get('ts', 0))
                
                market_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(data.get('open24h', 0)),
                    'high': float(data.get('high24h', 0)),
                    'low': float(data.get('low24h', 0)),
                    'close': float(data.get('last', 0)),
                    'volume': float(data.get('vol24h', 0)),
                    'exchange': exchange.value
                }
            
            # 存储到数据库
            await self.db_manager.write_market_data(
                symbol=market_data['symbol'],
                timestamp=market_data['timestamp'],
                ohlcv={
                    'open': market_data['open'],
                    'high': market_data['high'],
                    'low': market_data['low'],
                    'close': market_data['close'],
                    'volume': market_data['volume']
                }
            )
            
            # 调用回调函数
            for callback in self.data_callbacks:
                await callback('ticker', market_data)
                
        except Exception as e:
            self.logger.error(f"处理ticker数据失败: {e}")
    
    async def _handle_depth_data(self, exchange: DataSource, data: Dict[str, Any]):
        """处理深度数据"""
        # 简化处理，只记录买一卖一价格
        try:
            if exchange == DataSource.BINANCE:
                symbol = data.get('s', '').upper()
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
            elif exchange == DataSource.OKEX:
                symbol = data.get('instId', '').replace('-', '').upper()
                bids = data.get('bids', [])
                asks = data.get('asks', [])
            
            if bids and asks:
                depth_data = {
                    'symbol': symbol,
                    'timestamp': TimeUtils.now_timestamp(),
                    'bid_price': float(bids[0][0]),
                    'bid_size': float(bids[0][1]),
                    'ask_price': float(asks[0][0]),
                    'ask_size': float(asks[0][1]),
                    'exchange': exchange.value
                }
                
                # 调用回调函数
                for callback in self.data_callbacks:
                    await callback('depth', depth_data)
                    
        except Exception as e:
            self.logger.error(f"处理depth数据失败: {e}")
    
    async def _handle_trade_data(self, exchange: DataSource, data: Dict[str, Any]):
        """处理实时交易数据"""
        try:
            if exchange == DataSource.BINANCE:
                symbol = data.get('s', '').upper()
                trade_data = {
                    'symbol': symbol,
                    'timestamp': int(data.get('T', 0)),
                    'price': float(data.get('p', 0)),
                    'quantity': float(data.get('q', 0)),
                    'is_buyer_maker': data.get('m', False),
                    'exchange': exchange.value
                }
                
            elif exchange == DataSource.OKEX:
                symbol = data.get('instId', '').replace('-', '').upper()
                trade_data = {
                    'symbol': symbol,
                    'timestamp': int(data.get('ts', 0)),
                    'price': float(data.get('px', 0)),
                    'quantity': float(data.get('sz', 0)),
                    'is_buyer_maker': data.get('side') == 'sell',
                    'exchange': exchange.value
                }
            
            # 调用回调函数
            for callback in self.data_callbacks:
                await callback('trade', trade_data)
                
        except Exception as e:
            self.logger.error(f"处理trade数据失败: {e}")
    
    async def _start_rest_polling(self, symbols: List[str], exchanges: List[DataSource]):
        """启动REST轮询备份"""
        poll_interval = self.config.get("collector.rest_poll_interval", 60)
        
        while self.is_running:
            try:
                for exchange in exchanges:
                    # 检查WebSocket连接状态
                    if self.connection_states.get(exchange.value) != ConnectionState.CONNECTED:
                        self.logger.info(f"WebSocket断开，使用REST API获取 {exchange.value} 数据")
                        await self._fetch_rest_data(exchange, symbols)
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(f"REST轮询失败: {e}")
                await asyncio.sleep(poll_interval)
    
    @RetryDecorator(max_retries=3, delay=2.0)
    async def _fetch_rest_data(self, exchange: DataSource, symbols: List[str]):
        """通过REST API获取数据"""
        config = self.exchanges[exchange]
        base_url = config["rest_url"]
        connector = aiohttp.TCPConnector()
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            # 添加代理
            proxy = self.proxy_config['http'] if self.proxy_config['enabled'] else None
            
            for symbol in symbols:
                try:
                    if exchange == DataSource.BINANCE:
                        url = f"{base_url}ticker/24hr?symbol={symbol.upper()}"
                        
                        async with session.get(url, proxy=proxy) as response:
                            if response.status == 200:
                                data = await response.json()
                                await self._handle_ticker_data(exchange, data)
                            else:
                                self.logger.error(f"REST API请求失败: {response.status}")
                    
                    # 避免触发限流
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"获取REST数据失败 {symbol}: {e}")
    
    async def _monitor_data_quality(self):
        """数据质量监控"""
        monitor_interval = self.config.get("collector.quality_monitor_interval", 300)
        
        while self.is_running:
            try:
                await asyncio.sleep(monitor_interval)
                
                # 检查连接状态
                for exchange_name, state in self.connection_states.items():
                    if state != ConnectionState.CONNECTED:
                        self.logger.warning(f"{exchange_name} 连接状态异常: {state.value}")
                
                # 检查数据延迟
                # TODO: 实现数据时效性检查
                
            except Exception as e:
                self.logger.error(f"数据质量监控失败: {e}")
    
    def add_data_callback(self, callback: Callable):
        """添加数据回调函数"""
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable):
        """移除数据回调函数"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    async def stop(self):
        """停止数据采集"""
        self.logger.info("正在停止数据采集...")
        self.is_running = False
        
        # 关闭WebSocket连接
        for exchange_name, websocket in self.websockets.items():
            try:
                await websocket.close()
            except Exception as e:
                self.logger.error(f"关闭WebSocket失败 {exchange_name}: {e}")
        
        # 取消所有任务
        for task in self.running_tasks:
            task.cancel()
        
        # 等待任务完成
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        # 关闭数据库连接
        self.db_manager.close()
        
        self.logger.info("数据采集已停止")
    
    def get_connection_status(self) -> Dict[str, str]:
        """获取连接状态"""
        return {name: state.value for name, state in self.connection_states.items()}