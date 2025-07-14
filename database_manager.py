# database_manager.py - 时序数据库管理
# =============================================================================

import asyncio
import aiohttp
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from logger import TradingLogger
from utils import ConfigManager

class InfluxDBManager:
    """InfluxDB时序数据库管理器"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.logger = TradingLogger().get_logger("DATABASE")
        
        # 数据库配置
        self.host = config.get("database.host", "localhost")
        self.port = config.get("database.port", 8086)
        self.database = config.get("database.database", "ml2025_trading")
        self.username = config.get("database.username", "")
        self.password = config.get("database.password", "")
        self.token = config.get("database.token", "")
        self.org = config.get("database.org", "stark4")
        self.bucket = config.get("database.bucket", self.database)  # For InfluxDB 2.x
        
        self._init_connection()
    
    def _init_connection(self):
        """初始化数据库连接"""
        try:
            url = f"http://{self.host}:{self.port}"
            
            if self.token:
                # InfluxDB 2.x with token
                self.client = InfluxDBClient(
                    url=url,
                    token=self.token,
                    org=self.org
                )
            else:
                # InfluxDB 1.x with username/password
                self.client = InfluxDBClient(
                    url=url,
                    username=self.username,
                    password=self.password,
                    database=self.database
                )
            
            self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            self.logger.info("InfluxDB连接初始化成功")
            
        except Exception as e:
            self.logger.error(f"InfluxDB连接初始化失败: {e}")
            raise
    
    async def write_market_data(
        self,
        symbol: str,
        timestamp: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        measurement: str = "market_data"
    ):
        """写入市场数据"""
        try:
            point = Point(measurement) \
                .tag("symbol", symbol) \
                .field("open", float(open_price)) \
                .field("high", float(high_price)) \
                .field("low", float(low_price)) \
                .field("close", float(close_price)) \
                .field("volume", float(volume)) \
                .time(timestamp * 1000000)  # 转换为纳秒
            
            # Always include org parameter for InfluxDB 2.x
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
        except Exception as e:
            self.logger.error(f"写入市场数据失败: {e}")
            raise
    
    async def write_trade_signal(
        self,
        symbol: str,
        timestamp: int,
        action: str,
        size: float,
        price: float,
        confidence: float,
        model_type: str,
        measurement: str = "trade_signals"
    ):
        """写入交易信号"""
        try:
            point = Point(measurement) \
                .tag("symbol", symbol) \
                .tag("action", action) \
                .tag("model_type", model_type) \
                .field("size", float(size)) \
                .field("price", float(price)) \
                .field("confidence", float(confidence)) \
                .time(timestamp * 1000000)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
        except Exception as e:
            self.logger.error(f"写入交易信号失败: {e}")
            raise
    
    async def write_model_metrics(
        self,
        model_name: str,
        timestamp: int,
        metrics: Dict[str, float],
        measurement: str = "model_metrics"
    ):
        """写入模型指标"""
        try:
            point = Point(measurement) \
                .tag("model_name", model_name) \
                .time(timestamp * 1000000)
            
            for metric_name, value in metrics.items():
                point = point.field(metric_name, float(value))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
        except Exception as e:
            self.logger.error(f"写入模型指标失败: {e}")
            raise
    
    async def write_trade_data(
        self,
        symbol: str,
        timestamp: int,
        price: float,
        quantity: float,
        is_buyer_maker: bool,
        measurement: str = "trades"
    ):
        """写入交易数据"""
        try:
            point = Point(measurement) \
                .tag("symbol", symbol) \
                .tag("side", "sell" if is_buyer_maker else "buy") \
                .field("price", float(price)) \
                .field("quantity", float(quantity)) \
                .time(timestamp * 1000000)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
        except Exception as e:
            self.logger.error(f"写入交易数据失败: {e}")
            raise
    
    async def query_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        measurement: str = "market_data"
    ) -> pd.DataFrame:
        """查询市场数据"""
        try:
            bucket = self.bucket if self.token else self.database
            
            query = f'''
            from(bucket: "{bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            if self.token:
                result = self.query_api.query_data_frame(query, org=self.org)
            else:
                result = self.query_api.query_data_frame(query)
            
            if not result.empty:
                result['_time'] = pd.to_datetime(result['_time'])
                result = result.set_index('_time')
                result = result.sort_index()
            
            return result
            
        except Exception as e:
            self.logger.error(f"查询市场数据失败: {e}")
            raise
    
    async def get_latest_price(self, symbol: str, measurement: str = "market_data") -> Optional[float]:
        """获取最新价格"""
        try:
            bucket = self.bucket if self.token else self.database
            
            query = f'''
            from(bucket: "{bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r._field == "close")
                |> last()
            '''
            
            if self.token:
                result = self.query_api.query_data_frame(query, org=self.org)
            else:
                result = self.query_api.query_data_frame(query)
            
            if not result.empty:
                return float(result['_value'].iloc[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取最新价格失败: {e}")
            return None
    
    async def calculate_technical_indicators(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        indicators: List[str]
    ) -> Dict[str, np.ndarray]:
        """计算技术指标"""
        try:
            # 获取基础数据
            df = await self.query_market_data(symbol, start_time, end_time)
            
            if df.empty:
                return {}
            
            results = {}
            
            for indicator in indicators:
                if indicator == "sma_20":
                    results[indicator] = df['close'].rolling(window=20).mean().values
                elif indicator == "ema_12":
                    results[indicator] = df['close'].ewm(span=12).mean().values
                elif indicator == "rsi_14":
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    results[indicator] = (100 - (100 / (1 + rs))).values
                elif indicator == "macd":
                    ema_12 = df['close'].ewm(span=12).mean()
                    ema_26 = df['close'].ewm(span=26).mean()
                    results[indicator] = (ema_12 - ema_26).values
                elif indicator == "bollinger_upper":
                    sma_20 = df['close'].rolling(window=20).mean()
                    std_20 = df['close'].rolling(window=20).std()
                    results[indicator] = (sma_20 + 2 * std_20).values
                elif indicator == "bollinger_lower":
                    sma_20 = df['close'].rolling(window=20).mean()
                    std_20 = df['close'].rolling(window=20).std()
                    results[indicator] = (sma_20 - 2 * std_20).values
            
            return results
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return {}
    
    async def get_model_performance_history(
        self,
        model_name: str,
        days: int = 30
    ) -> pd.DataFrame:
        """获取模型性能历史"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            bucket = self.bucket if self.token else self.database
            
            query = f'''
            from(bucket: "{bucket}")
                |> range(start: {start_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "model_metrics")
                |> filter(fn: (r) => r.model_name == "{model_name}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            if self.token:
                result = self.query_api.query_data_frame(query, org=self.org)
            else:
                result = self.query_api.query_data_frame(query)
            
            if not result.empty:
                result['_time'] = pd.to_datetime(result['_time'])
                result = result.set_index('_time')
                result = result.sort_index()
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取模型性能历史失败: {e}")
            return pd.DataFrame()
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.write_api:
                self.write_api.close()
            if self.client:
                self.client.close()
            self.logger.info("数据库连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭数据库连接失败: {e}")


class DatabaseManager:
    """数据库管理器主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("DATABASE_MANAGER")
        
        # 根据配置选择数据库类型
        db_type = self.config.get("database.type", "influxdb")
        
        if db_type == "influxdb":
            self.db = InfluxDBManager(self.config)
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    async def write_market_data(self, symbol: str, timestamp: int, ohlcv: Dict[str, float]):
        """写入市场数据的便捷方法"""
        await self.db.write_market_data(
            symbol=symbol,
            timestamp=timestamp,
            open_price=ohlcv['open'],
            high_price=ohlcv['high'],
            low_price=ohlcv['low'],
            close_price=ohlcv['close'],
            volume=ohlcv['volume']
        )
    
    async def get_training_data(
        self,
        symbol: str,
        days: int = 30,
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """获取训练数据"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 获取基础市场数据
            market_data = await self.db.query_market_data(symbol, start_time, end_time)
            
            result = {
                "market_data": market_data,
                "indicators": {}
            }
            
            # 计算技术指标
            if indicators:
                result["indicators"] = await self.db.calculate_technical_indicators(
                    symbol, start_time, end_time, indicators
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取训练数据失败: {e}")
            return {"market_data": pd.DataFrame(), "indicators": {}}
    
    def close(self):
        """关闭数据库连接"""
        self.db.close()