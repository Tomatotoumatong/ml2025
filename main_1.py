#!/usr/bin/env python3
# main_1.py - 第一阶段：数据采集与清洗模块集成
# =============================================================================
# 核心功能：
# 1. 异步数据采集（Binance/Deribit）
# 2. 数据清洗和标准化
# 3. 数据存储到数据库
# 4. 定时任务调度
# =============================================================================

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import schedule
import time
from pathlib import Path

# 导入项目模块
from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from database_manager import DatabaseManager
from data_collector import MarketDataCollector, DataSource
from data_cleaner import DataCleaner


class DataPipeline:
    """第一阶段数据管道"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("DATA_PIPELINE")
        
        # 初始化组件
        self.db_manager = DatabaseManager(config_path)
        self.data_collector = MarketDataCollector(config_path)
        self.data_cleaner = DataCleaner(config_path)
        
        # 运行状态
        self.is_running = False
        self.collection_task = None
        
        # 数据缓冲区
        self.data_buffer = {
            'ticker': [],
            'trade': [],
            'depth': []
        }
        self.buffer_size = self.config.get("pipeline.buffer_size", 1000)
        self.flush_interval = self.config.get("pipeline.flush_interval", 60)
        
        # 配置
        self.symbols = self.config.get("trading.symbols", ["BTCUSDT", "ETHUSDT"])
        self.exchanges = [DataSource.BINANCE]  # 可扩展到其他交易所
        
        # 统计信息
        self.stats = {
            'total_received': 0,
            'total_cleaned': 0,
            'total_stored': 0,
            'errors': 0,
            'last_update': TimeUtils.now_timestamp()
        }
        
        self.logger.info("数据管道初始化完成")
    
    async def start(self):
        """启动数据管道"""
        try:
            self.logger.info("启动数据管道...")
            self.is_running = True
            
            # 注册数据回调
            self.data_collector.add_data_callback(self._handle_market_data)
            
            # 启动数据采集
            await self.data_collector.start(self.symbols, self.exchanges)
            
            # 启动定时任务
            self._start_scheduled_tasks()
            
            # 启动缓冲区刷新任务
            self.collection_task = asyncio.create_task(self._buffer_flush_loop())
            
            self.logger.info(f"数据管道已启动，监控交易对: {self.symbols}")
            
        except Exception as e:
            self.logger.error(f"启动数据管道失败: {e}")
            raise
    
    async def stop(self):
        """停止数据管道"""
        self.logger.info("停止数据管道...")
        self.is_running = False
        
        # 停止数据采集
        await self.data_collector.stop()
        
        # 刷新剩余数据
        await self._flush_buffers()
        
        # 取消任务
        if self.collection_task:
            self.collection_task.cancel()
        
        # 关闭数据库连接
        self.db_manager.close()
        
        self.logger.info("数据管道已停止")
    
    async def _handle_market_data(self, data_type: str, data: Dict[str, Any]):
        """处理接收到的市场数据"""
        try:
            self.stats['total_received'] += 1
            
            # 添加到缓冲区
            if data_type in self.data_buffer:
                self.data_buffer[data_type].append(data)
                
                # 检查缓冲区大小
                if len(self.data_buffer[data_type]) >= self.buffer_size:
                    await self._process_buffer(data_type)
            
        except Exception as e:
            self.logger.error(f"处理市场数据失败: {e}")
            self.stats['errors'] += 1
    
    async def _process_buffer(self, data_type: str):
        """处理缓冲区数据"""
        if not self.data_buffer[data_type]:
            return
        
        try:
            # 转换为DataFrame
            df = pd.DataFrame(self.data_buffer[data_type])
            
            # 按交易对分组处理
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                
                # 数据清洗
                if data_type == 'ticker':
                    cleaned_data = await self._clean_ticker_data(symbol_data, symbol)
                elif data_type == 'trade':
                    cleaned_data = await self._clean_trade_data(symbol_data, symbol)
                elif data_type == 'depth':
                    cleaned_data = await self._clean_depth_data(symbol_data, symbol)
                else:
                    continue
                
                # 存储到数据库
                if cleaned_data is not None and not cleaned_data.empty:
                    await self._store_data(cleaned_data, data_type, symbol)
            
            # 清空缓冲区
            self.data_buffer[data_type] = []
            
        except Exception as e:
            self.logger.error(f"处理缓冲区失败 [{data_type}]: {e}")
            self.stats['errors'] += 1
    
    async def _clean_ticker_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """清洗ticker数据"""
        try:
            # 重命名列以匹配清洗器期望的格式
            df_clean = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # 执行清洗
            cleaned_data, quality_report = self.data_cleaner.clean_market_data(
                df_clean,
                symbol,
                detect_outliers=True,
                handle_missing=True,
                normalize_data=False  # 原始数据不标准化
            )
            
            self.stats['total_cleaned'] += len(cleaned_data)
            
            # 记录数据质量
            if quality_report['data_quality_score'] < 0.8:
                self.logger.warning(
                    f"{symbol} 数据质量较低: {quality_report['data_quality_score']:.2f}"
                )
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"清洗ticker数据失败: {e}")
            return None
    
    async def _clean_trade_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """清洗交易数据"""
        try:
            # 基础清洗
            df_clean = df.copy()
            
            # 移除重复数据
            df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='last')
            
            # 移除无效价格
            df_clean = df_clean[
                (df_clean['price'] > 0) & 
                (df_clean['quantity'] > 0)
            ]
            
            # 按时间排序
            df_clean = df_clean.sort_values('timestamp')
            
            self.stats['total_cleaned'] += len(df_clean)
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"清洗trade数据失败: {e}")
            return None
    
    async def _clean_depth_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """清洗深度数据"""
        try:
            df_clean = df.copy()
            
            # 移除无效数据
            df_clean = df_clean[
                (df_clean['bid_price'] > 0) & 
                (df_clean['ask_price'] > 0) &
                (df_clean['bid_price'] < df_clean['ask_price'])  # 买价必须小于卖价
            ]
            
            # 移除重复时间戳
            df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='last')
            
            self.stats['total_cleaned'] += len(df_clean)
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"清洗depth数据失败: {e}")
            return None
    
    async def _store_data(self, df: pd.DataFrame, data_type: str, symbol: str):
        """存储数据到数据库"""
        try:
            if data_type == 'ticker':
                # 存储OHLCV数据
                for _, row in df.iterrows():
                    await self.db_manager.write_market_data(
                        symbol=symbol,
                        timestamp=int(row['timestamp']),
                        ohlcv={
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        }
                    )
            
            elif data_type == 'trade':
                # 存储交易数据
                for _, row in df.iterrows():
                    await self.db_manager.db.write_trade_data(
                        symbol=symbol,
                        timestamp=int(row['timestamp']),
                        price=row['price'],
                        quantity=row['quantity'],
                        is_buyer_maker=row.get('is_buyer_maker', False)
                    )
            
            elif data_type == 'depth':
                # 存储深度数据
                for _, row in df.iterrows():
                    # 这里可以扩展database_manager添加depth数据存储方法
                    pass
            
            self.stats['total_stored'] += len(df)
            self.logger.debug(f"存储 {len(df)} 条 {data_type} 数据 [{symbol}]")
            
        except Exception as e:
            self.logger.error(f"存储数据失败: {e}")
            self.stats['errors'] += 1
    
    async def _flush_buffers(self):
        """刷新所有缓冲区"""
        self.logger.info("刷新数据缓冲区...")
        
        for data_type in self.data_buffer:
            if self.data_buffer[data_type]:
                await self._process_buffer(data_type)
    
    async def _buffer_flush_loop(self):
        """定期刷新缓冲区"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffers()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"缓冲区刷新循环错误: {e}")
    
    def _start_scheduled_tasks(self):
        """启动定时任务"""
        # 每小时生成统计报告
        schedule.every().hour.do(self._generate_hourly_report)
        
        # 每天清理历史数据
        schedule.every().day.at("00:00").do(self._cleanup_old_data)
        
        # 每30分钟检查数据质量
        schedule.every(30).minutes.do(self._check_data_quality)
        
        # 启动调度线程
        asyncio.create_task(self._run_schedule())
    
    async def _run_schedule(self):
        """运行调度任务"""
        while self.is_running:
            try:
                schedule.run_pending()
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"调度任务错误: {e}")
    
    def _generate_hourly_report(self):
        """生成小时报告"""
        try:
            report = {
                'timestamp': TimeUtils.now_timestamp(),
                'total_received': self.stats['total_received'],
                'total_cleaned': self.stats['total_cleaned'],
                'total_stored': self.stats['total_stored'],
                'errors': self.stats['errors'],
                'success_rate': (
                    (self.stats['total_stored'] / self.stats['total_received'] * 100)
                    if self.stats['total_received'] > 0 else 0
                )
            }
            
            self.logger.info(
                f"📊 小时报告 - "
                f"接收: {report['total_received']}, "
                f"清洗: {report['total_cleaned']}, "
                f"存储: {report['total_stored']}, "
                f"成功率: {report['success_rate']:.2f}%"
            )
            
            # 重置计数器
            self.stats.update({
                'total_received': 0,
                'total_cleaned': 0,
                'total_stored': 0,
                'errors': 0,
                'last_update': TimeUtils.now_timestamp()
            })
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
    
    def _cleanup_old_data(self):
        """清理历史数据"""
        try:
            retention_days = self.config.get("database.retention_days", 30)
            self.logger.info(f"清理 {retention_days} 天前的历史数据...")
            
            # 这里可以调用数据库管理器的清理方法
            # self.db_manager.cleanup_old_data(retention_days)
            
        except Exception as e:
            self.logger.error(f"清理历史数据失败: {e}")
    
    def _check_data_quality(self):
        """检查数据质量"""
        try:
            # 检查最近的数据质量
            for symbol in self.symbols:
                # 这里可以实现数据质量检查逻辑
                pass
                
        except Exception as e:
            self.logger.error(f"检查数据质量失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


async def main():
    """主函数"""
    # 创建数据管道
    pipeline = DataPipeline("config.yaml")
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n正在停止数据管道...")
        asyncio.create_task(pipeline.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动管道
        await pipeline.start()
        
        # 保持运行
        while pipeline.is_running:
            await asyncio.sleep(10)
            
            # 打印状态
            stats = pipeline.get_stats()
            print(f"\r📈 状态 - 接收: {stats['total_received']}, "
                  f"存储: {stats['total_stored']}, "
                  f"错误: {stats['errors']}", end='')
    
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    # 确保配置文件存在
    config_path = Path("config.yaml")
    if not config_path.exists():
        # 创建默认配置
        default_config = """
database:
  type: influxdb
  host: localhost
  port: 8086
  database: ml2025_trading
  username: ""
  password: ""
  retention_days: 30

trading:
  symbols:
    - BTCUSDT
    - ETHUSDT
  exchanges:
    - binance

collector:
  max_reconnect_attempts: 10
  reconnect_delay: 5
  heartbeat_interval: 30
  rest_poll_interval: 60

network:
  proxy_enabled: true
  proxy_http: "http://127.0.0.1:7890"
  proxy_https: "http://127.0.0.1:7890"

cleaner:
  outlier_method: modified_zscore
  outlier_threshold: 3.5
  imputation_method: linear
  scaling_method: robust
  outlier_strategy: cap

pipeline:
  buffer_size: 1000
  flush_interval: 60

logging:
  level: INFO
  max_bytes: 10485760
  backup_count: 5
"""
        config_path.write_text(default_config)
        print("已创建默认配置文件 config.yaml")
    
    # 运行主程序
    asyncio.run(main())