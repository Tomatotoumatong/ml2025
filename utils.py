
# utils.py - 通用工具函数
# =============================================================================

import time
import json
import hashlib
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Union, Dict, Any, List, Optional, Callable
from pathlib import Path
import yaml
import numpy as np
import pandas as pd


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = self._get_default_config()
                self.save_config()
        except Exception as e:
            print(f"配置加载失败: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "database": {
                "type": "influxdb",
                "host": "localhost",
                "port": 8086,
                "database": "stark4_trading",
                "username": "",
                "password": ""
            },
            "trading": {
                "max_position_size": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_daily_trades": 50
            },
            "models": {
                "retrain_threshold": 0.45,
                "validation_window": 1000,
                "experience_replay_size": 10000
            },
            "notifications": {
                "telegram_enabled": False,
                "telegram_token": "",
                "telegram_chat_id": ""
            }
        }
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"配置保存失败: {e}")
    
    def get(self, key: str, default=None):
        """获取配置值（支持点分隔路径）"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值（支持点分隔路径）"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()


class TimeUtils:
    """时间工具类"""
    
    @staticmethod
    def now_timestamp() -> int:
        """获取当前时间戳（毫秒）"""
        return int(time.time() * 1000)
    
    @staticmethod
    def timestamp_to_datetime(timestamp: int) -> datetime:
        """时间戳转换为datetime对象"""
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """datetime对象转换为时间戳（毫秒）"""
        return int(dt.timestamp() * 1000)
    
    @staticmethod
    def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """格式化datetime对象"""
        return dt.strftime(fmt)
    
    @staticmethod
    def parse_datetime(date_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """解析日期字符串"""
        return datetime.strptime(date_str, fmt)


class PriceUtils:
    """价格工具类"""
    
    @staticmethod
    def to_decimal(value: Union[str, float, int]) -> Decimal:
        """转换为Decimal类型"""
        return Decimal(str(value))
    
    @staticmethod
    def round_price(price: Union[str, float, Decimal], precision: int = 8) -> Decimal:
        """四舍五入价格"""
        price_decimal = PriceUtils.to_decimal(price)
        return price_decimal.quantize(
            Decimal('0.' + '0' * precision),
            rounding=ROUND_HALF_UP
        )
    
    @staticmethod
    def calculate_percentage_change(old_price: float, new_price: float) -> float:
        """计算百分比变化"""
        if old_price == 0:
            return 0
        return ((new_price - old_price) / old_price) * 100
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """计算仓位大小"""
        risk_amount = account_balance * risk_per_trade
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        return risk_amount / price_diff


class DataUtils:
    """数据工具类"""
    
    @staticmethod
    def normalize_features(data: np.ndarray) -> np.ndarray:
        """特征标准化"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # 避免除零
        return (data - mean) / std
    
    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
        """创建序列数据"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量"""
        quality_report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        return quality_report
    
    @staticmethod
    def hash_data(data: Any) -> str:
        """计算数据哈希值"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()


class RetryDecorator:
    """重试装饰器"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
    
    def __call__(self, func: Callable):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = self.delay
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.max_retries:
                        time.sleep(current_delay)
                        current_delay *= self.backoff
                    else:
                        raise last_exception
            
            return wrapper
        
        return wrapper


async def async_retry(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """异步重试函数"""
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            last_exception = e
            
            if attempt < self.max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                raise last_exception

