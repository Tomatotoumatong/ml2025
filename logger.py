
# logger.py - 统一日志系统
# =============================================================================

import logging
import logging.handlers
import asyncio
import threading
import queue
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class AsyncLogHandler(logging.Handler):
    """异步日志处理器"""
    
    def __init__(self, handler: logging.Handler):
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """后台线程处理日志记录"""
        while True:
            try:
                record = self.queue.get(timeout=1)
                if record is None:
                    break
                self.handler.handle(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"日志处理错误: {e}", file=sys.stderr)
    
    def emit(self, record):
        """发送日志记录到队列"""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            print("日志队列已满，丢弃日志记录", file=sys.stderr)
    
    def close(self):
        """关闭处理器"""
        self.queue.put(None)
        self.thread.join(timeout=5)
        self.handler.close()
        super().close()


class TradingLogger:
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建系统默认logger
        self.setup_logger(
            name="STARK4",
            level=LogLevel.INFO,
            file_name="stark4_system.log"
        )
    
    def setup_logger(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        file_name: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_async: bool = True
    ) -> logging.Logger:
        """
        设置日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            file_name: 日志文件名
            max_bytes: 单个日志文件最大字节数
            backup_count: 备份文件数量
            enable_console: 是否启用控制台输出
            enable_async: 是否启用异步处理
        
        Returns:
            配置好的日志器
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level.value)
        logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        if file_name:
            file_path = self.log_dir / file_name
            file_handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            
            if enable_async:
                file_handler = AsyncLogHandler(file_handler)
            
            logger.addHandler(file_handler)
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            
            if enable_async:
                console_handler = AsyncLogHandler(console_handler)
            
            logger.addHandler(console_handler)
        
        self.loggers[name] = logger
        return logger
    
    def get_logger(self, name: str = "STARK4") -> logging.Logger:
        """获取指定名称的日志器"""
        if name not in self.loggers:
            return self.setup_logger(name)
        return self.loggers[name]
    
    def log_trade_signal(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        confidence: float,
        model_type: str
    ):
        """记录交易信号"""
        trade_logger = self.get_logger("TRADE_SIGNAL")
        signal_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "confidence": confidence,
            "model_type": model_type
        }
        trade_logger.info(f"交易信号: {json.dumps(signal_data, ensure_ascii=False)}")
    
    def log_model_performance(
        self,
        model_name: str,
        accuracy: float,
        sharpe_ratio: float,
        max_drawdown: float
    ):
        """记录模型性能"""
        perf_logger = self.get_logger("MODEL_PERFORMANCE")
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "accuracy": accuracy,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
        perf_logger.info(f"模型性能: {json.dumps(perf_data, ensure_ascii=False)}")
    
    def log_system_error(self, module: str, error: Exception, context: Dict[str, Any] = None):
        """记录系统错误"""
        error_logger = self.get_logger("SYSTEM_ERROR")
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        error_logger.error(f"系统错误: {json.dumps(error_data, ensure_ascii=False)}")

