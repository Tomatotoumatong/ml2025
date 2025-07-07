# system_monitor.py - 系统监控模块
# =============================================================================
# 核心职责：
# 1. 系统资源监控（CPU、内存、磁盘、网络）
# 2. 应用性能监控（延迟、吞吐量、错误率）
# 3. 交易性能监控（成功率、盈亏、滑点）
# 4. 异常检测和告警
# 5. 健康检查和自动恢复
# =============================================================================

import psutil
import asyncio
import threading
import queue
import time
import json
import os
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from database_manager import DatabaseManager


class MetricType(Enum):
    """指标类型枚举"""
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_NETWORK = "system_network"
    APP_LATENCY = "app_latency"
    APP_THROUGHPUT = "app_throughput"
    APP_ERROR_RATE = "app_error_rate"
    TRADE_SUCCESS_RATE = "trade_success_rate"
    TRADE_PNL = "trade_pnl"
    TRADE_SLIPPAGE = "trade_slippage"
    MODEL_ACCURACY = "model_accuracy"
    MODEL_LATENCY = "model_latency"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class Metric:
    """监控指标"""
    name: str
    type: MetricType
    value: float
    timestamp: int
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


@dataclass
class Alert:
    """告警信息"""
    metric_name: str
    level: AlertLevel
    message: str
    current_value: float
    threshold: float
    timestamp: int
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'level': self.level.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'resolved': self.resolved
        }


@dataclass
class HealthCheck:
    """健康检查结果"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = field(default_factory=TimeUtils.now_timestamp)


class SystemMonitor:
    """
    系统监控器
    
    核心功能：
    1. 实时收集系统和应用指标
    2. 异常检测和告警
    3. 健康检查
    4. 性能分析
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("SYSTEM_MONITOR")
        
        # 初始化组件
        self.db_manager = DatabaseManager(config_path)
        
        # 监控配置
        self.collection_interval = self.config.get("monitor.collection_interval", 5)  # 秒
        self.metric_retention = self.config.get("monitor.metric_retention", 3600)  # 秒
        
        # 告警阈值配置
        self.thresholds = self._init_thresholds()
        
        # 指标存储
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics = {}
        
        # 告警管理
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # 健康检查
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_check_callbacks: List[Callable] = []
        
        # 性能统计
        self.performance_stats = {
            'latency_percentiles': {},
            'error_counts': defaultdict(int),
            'success_counts': defaultdict(int)
        }
        
        # 监控线程
        self.monitor_thread = None
        self.is_running = False
        self.metrics_queue = queue.Queue()
        
        # 组件注册
        self.registered_components: Set[str] = set()
        self.component_monitors: Dict[str, Callable] = {}
        
        self.logger.info("系统监控器初始化完成")
    
    def _init_thresholds(self) -> Dict[MetricType, Dict[str, float]]:
        """初始化告警阈值"""
        return {
            MetricType.SYSTEM_CPU: {
                'warning': self.config.get("monitor.thresholds.cpu_warning", 70),
                'error': self.config.get("monitor.thresholds.cpu_error", 85),
                'critical': self.config.get("monitor.thresholds.cpu_critical", 95)
            },
            MetricType.SYSTEM_MEMORY: {
                'warning': self.config.get("monitor.thresholds.memory_warning", 70),
                'error': self.config.get("monitor.thresholds.memory_error", 85),
                'critical': self.config.get("monitor.thresholds.memory_critical", 95)
            },
            MetricType.SYSTEM_DISK: {
                'warning': self.config.get("monitor.thresholds.disk_warning", 80),
                'error': self.config.get("monitor.thresholds.disk_error", 90),
                'critical': self.config.get("monitor.thresholds.disk_critical", 95)
            },
            MetricType.APP_ERROR_RATE: {
                'warning': self.config.get("monitor.thresholds.error_rate_warning", 0.01),
                'error': self.config.get("monitor.thresholds.error_rate_error", 0.05),
                'critical': self.config.get("monitor.thresholds.error_rate_critical", 0.1)
            },
            MetricType.APP_LATENCY: {
                'warning': self.config.get("monitor.thresholds.latency_warning", 100),
                'error': self.config.get("monitor.thresholds.latency_error", 500),
                'critical': self.config.get("monitor.thresholds.latency_critical", 1000)
            },
            MetricType.TRADE_SUCCESS_RATE: {
                'warning': self.config.get("monitor.thresholds.trade_success_warning", 0.95),
                'error': self.config.get("monitor.thresholds.trade_success_error", 0.9),
                'critical': self.config.get("monitor.thresholds.trade_success_critical", 0.8)
            }
        }
    
    def start(self):
        """启动监控"""
        if self.is_running:
            self.logger.warning("监控器已在运行")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                self._collect_system_metrics()
                
                # 收集组件指标
                self._collect_component_metrics()
                
                # 处理队列中的指标
                self._process_metric_queue()
                
                # 检查告警
                self._check_alerts()
                
                # 执行健康检查
                self._perform_health_checks()
                
                # 清理过期数据
                self._cleanup_old_metrics()
                
                # 等待下一个周期
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        timestamp = TimeUtils.now_timestamp()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric(
            name="cpu_usage",
            type=MetricType.SYSTEM_CPU,
            value=cpu_percent,
            timestamp=timestamp
        )
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.record_metric(
            name="memory_usage",
            type=MetricType.SYSTEM_MEMORY,
            value=memory.percent,
            timestamp=timestamp
        )
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        self.record_metric(
            name="disk_usage",
            type=MetricType.SYSTEM_DISK,
            value=disk.percent,
            timestamp=timestamp
        )
        
        # 网络IO
        net_io = psutil.net_io_counters()
        self.record_metric(
            name="network_bytes_sent",
            type=MetricType.SYSTEM_NETWORK,
            value=net_io.bytes_sent,
            timestamp=timestamp,
            tags={'direction': 'sent'}
        )
        self.record_metric(
            name="network_bytes_recv",
            type=MetricType.SYSTEM_NETWORK,
            value=net_io.bytes_recv,
            timestamp=timestamp,
            tags={'direction': 'recv'}
        )
        
        # 进程级指标
        try:
            process = psutil.Process(os.getpid())
            process_info = process.as_dict(attrs=['cpu_percent', 'memory_percent', 'num_threads'])
            
            self.record_metric(
                name="process_cpu",
                type=MetricType.SYSTEM_CPU,
                value=process_info['cpu_percent'],
                timestamp=timestamp,
                tags={'process': 'stark4'}
            )
            
            self.record_metric(
                name="process_memory",
                type=MetricType.SYSTEM_MEMORY,
                value=process_info['memory_percent'],
                timestamp=timestamp,
                tags={'process': 'stark4'}
            )
            
            self.record_metric(
                name="process_threads",
                type=MetricType.SYSTEM_CPU,
                value=process_info['num_threads'],
                timestamp=timestamp,
                tags={'process': 'stark4'}
            )
            
        except Exception as e:
            self.logger.error(f"进程指标收集失败: {e}")
    
    def _collect_component_metrics(self):
        """收集组件指标"""
        for component_name, monitor_func in self.component_monitors.items():
            try:
                monitor_func()
            except Exception as e:
                self.logger.error(f"组件 {component_name} 指标收集失败: {e}")
    
    def _process_metric_queue(self):
        """处理指标队列"""
        processed = 0
        while not self.metrics_queue.empty() and processed < 100:
            try:
                metric = self.metrics_queue.get_nowait()
                self._store_metric(metric)
                processed += 1
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"处理指标失败: {e}")
    
    def _store_metric(self, metric: Metric):
        """存储指标"""
        # 添加到缓冲区
        key = f"{metric.type.value}:{metric.name}"
        self.metrics_buffer[key].append({
            'value': metric.value,
            'timestamp': metric.timestamp,
            'tags': metric.tags
        })
        
        # 更新当前值
        self.current_metrics[key] = metric.value
        
        # 异步写入数据库
        asyncio.create_task(self._write_metric_to_db(metric))
    
    async def _write_metric_to_db(self, metric: Metric):
        """写入指标到数据库"""
        try:
            await self.db_manager.db.write_model_metrics(
                model_name=f"system_{metric.type.value}",
                timestamp=metric.timestamp,
                metrics={metric.name: metric.value}
            )
        except Exception as e:
            self.logger.error(f"写入指标到数据库失败: {e}")
    
    def _check_alerts(self):
        """检查告警条件"""
        for metric_type, thresholds in self.thresholds.items():
            # 获取该类型的所有指标
            relevant_metrics = [
                (key, value) for key, value in self.current_metrics.items()
                if key.startswith(metric_type.value)
            ]
            
            for metric_key, current_value in relevant_metrics:
                # 检查各级别阈值
                for level_name in ['critical', 'error', 'warning']:
                    if level_name not in thresholds:
                        continue
                    
                    threshold = thresholds[level_name]
                    level = AlertLevel(level_name)
                    
                    # 根据指标类型判断是否触发告警
                    should_alert = self._should_trigger_alert(
                        metric_type, current_value, threshold
                    )
                    
                    if should_alert:
                        self._trigger_alert(
                            metric_name=metric_key,
                            level=level,
                            current_value=current_value,
                            threshold=threshold
                        )
                        break  # 只触发最高级别的告警
                    else:
                        # 如果之前有告警，现在恢复了
                        self._resolve_alert(metric_key)
    
    def _should_trigger_alert(self, metric_type: MetricType, 
                            current_value: float, threshold: float) -> bool:
        """判断是否应该触发告警"""
        # 对于成功率类指标，低于阈值触发告警
        if metric_type == MetricType.TRADE_SUCCESS_RATE:
            return current_value < threshold
        # 对于其他指标，高于阈值触发告警
        else:
            return current_value > threshold
    
    def _trigger_alert(self, metric_name: str, level: AlertLevel,
                      current_value: float, threshold: float):
        """触发告警"""
        # 检查是否已有同样的告警
        if metric_name in self.active_alerts:
            existing_alert = self.active_alerts[metric_name]
            # 如果级别升级，更新告警
            if level.value > existing_alert.level.value:
                existing_alert.level = level
                existing_alert.current_value = current_value
                existing_alert.timestamp = TimeUtils.now_timestamp()
            return
        
        # 创建新告警
        alert = Alert(
            metric_name=metric_name,
            level=level,
            message=f"{metric_name} 触发 {level.value} 级别告警: {current_value:.2f} > {threshold:.2f}",
            current_value=current_value,
            threshold=threshold,
            timestamp=TimeUtils.now_timestamp()
        )
        
        self.active_alerts[metric_name] = alert
        self.alert_history.append(alert)
        
        # 记录日志
        log_func = getattr(self.logger, level.value, self.logger.warning)
        log_func(alert.message)
        
        # 触发告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")
    
    def _resolve_alert(self, metric_name: str):
        """解决告警"""
        if metric_name in self.active_alerts:
            alert = self.active_alerts[metric_name]
            alert.resolved = True
            del self.active_alerts[metric_name]
            
            self.logger.info(f"告警已解决: {metric_name}")
    
    def _perform_health_checks(self):
        """执行健康检查"""
        # 系统健康检查
        self._check_system_health()
        
        # 应用健康检查
        self._check_application_health()
        
        # 组件健康检查
        for component in self.registered_components:
            self._check_component_health(component)
    
    def _check_system_health(self):
        """检查系统健康状态"""
        # CPU健康检查
        cpu_usage = self.current_metrics.get('system_cpu:cpu_usage', 0)
        cpu_status = HealthStatus.HEALTHY
        cpu_message = "CPU使用正常"
        
        if cpu_usage > 95:
            cpu_status = HealthStatus.CRITICAL
            cpu_message = f"CPU使用率过高: {cpu_usage:.1f}%"
        elif cpu_usage > 85:
            cpu_status = HealthStatus.UNHEALTHY
            cpu_message = f"CPU使用率高: {cpu_usage:.1f}%"
        elif cpu_usage > 70:
            cpu_status = HealthStatus.DEGRADED
            cpu_message = f"CPU使用率偏高: {cpu_usage:.1f}%"
        
        self.health_checks['system_cpu'] = HealthCheck(
            component='system_cpu',
            status=cpu_status,
            message=cpu_message,
            details={'usage': cpu_usage}
        )
        
        # 内存健康检查
        memory_usage = self.current_metrics.get('system_memory:memory_usage', 0)
        memory_status = HealthStatus.HEALTHY
        memory_message = "内存使用正常"
        
        if memory_usage > 95:
            memory_status = HealthStatus.CRITICAL
            memory_message = f"内存使用率过高: {memory_usage:.1f}%"
        elif memory_usage > 85:
            memory_status = HealthStatus.UNHEALTHY
            memory_message = f"内存使用率高: {memory_usage:.1f}%"
        elif memory_usage > 70:
            memory_status = HealthStatus.DEGRADED
            memory_message = f"内存使用率偏高: {memory_usage:.1f}%"
        
        self.health_checks['system_memory'] = HealthCheck(
            component='system_memory',
            status=memory_status,
            message=memory_message,
            details={'usage': memory_usage}
        )
    
    def _check_application_health(self):
        """检查应用健康状态"""
        # 基于错误率的健康检查
        error_rate = self.current_metrics.get('app_error_rate:error_rate', 0)
        
        if error_rate > 0.1:
            app_status = HealthStatus.CRITICAL
            app_message = f"错误率过高: {error_rate:.2%}"
        elif error_rate > 0.05:
            app_status = HealthStatus.UNHEALTHY
            app_message = f"错误率高: {error_rate:.2%}"
        elif error_rate > 0.01:
            app_status = HealthStatus.DEGRADED
            app_message = f"错误率偏高: {error_rate:.2%}"
        else:
            app_status = HealthStatus.HEALTHY
            app_message = "应用运行正常"
        
        self.health_checks['application'] = HealthCheck(
            component='application',
            status=app_status,
            message=app_message,
            details={'error_rate': error_rate}
        )
    
    def _check_component_health(self, component: str):
        """检查组件健康状态"""
        # 这里应该调用组件特定的健康检查方法
        # 暂时使用默认实现
        self.health_checks[component] = HealthCheck(
            component=component,
            status=HealthStatus.HEALTHY,
            message=f"{component} 运行正常"
        )
    
    def _cleanup_old_metrics(self):
        """清理过期指标"""
        current_time = TimeUtils.now_timestamp()
        cutoff_time = current_time - self.metric_retention * 1000
        
        for key, buffer in self.metrics_buffer.items():
            # 移除过期数据
            while buffer and buffer[0]['timestamp'] < cutoff_time:
                buffer.popleft()
    
    def record_metric(self, name: str, type: MetricType, value: float,
                     timestamp: Optional[int] = None, tags: Optional[Dict[str, str]] = None):
        """记录指标"""
        metric = Metric(
            name=name,
            type=type,
            value=value,
            timestamp=timestamp or TimeUtils.now_timestamp(),
            tags=tags or {}
        )
        
        self.metrics_queue.put(metric)
    
    def record_latency(self, operation: str, latency_ms: float):
        """记录延迟"""
        self.record_metric(
            name=f"latency_{operation}",
            type=MetricType.APP_LATENCY,
            value=latency_ms,
            tags={'operation': operation}
        )
        
        # 更新延迟统计
        if operation not in self.performance_stats['latency_percentiles']:
            self.performance_stats['latency_percentiles'][operation] = deque(maxlen=1000)
        
        self.performance_stats['latency_percentiles'][operation].append(latency_ms)
    
    def record_error(self, operation: str, error_type: str):
        """记录错误"""
        self.performance_stats['error_counts'][f"{operation}:{error_type}"] += 1
        
        # 计算错误率
        total_ops = (self.performance_stats['error_counts'][f"{operation}:{error_type}"] +
                    self.performance_stats['success_counts'][operation])
        
        if total_ops > 0:
            error_rate = self.performance_stats['error_counts'][f"{operation}:{error_type}"] / total_ops
            
            self.record_metric(
                name=f"error_rate_{operation}",
                type=MetricType.APP_ERROR_RATE,
                value=error_rate,
                tags={'operation': operation, 'error_type': error_type}
            )
    
    def record_success(self, operation: str):
        """记录成功操作"""
        self.performance_stats['success_counts'][operation] += 1
    
    def record_trade_metrics(self, symbol: str, success: bool, pnl: float, slippage: float):
        """记录交易指标"""
        # 交易成功率
        total_trades = (self.performance_stats['success_counts'][f'trade_{symbol}'] +
                       self.performance_stats['error_counts'][f'trade_{symbol}:failed'])
        
        if success:
            self.record_success(f'trade_{symbol}')
        else:
            self.record_error(f'trade_{symbol}', 'failed')
        
        if total_trades > 0:
            success_rate = self.performance_stats['success_counts'][f'trade_{symbol}'] / total_trades
            
            self.record_metric(
                name=f"trade_success_rate_{symbol}",
                type=MetricType.TRADE_SUCCESS_RATE,
                value=success_rate,
                tags={'symbol': symbol}
            )
        
        # 盈亏
        self.record_metric(
            name=f"trade_pnl_{symbol}",
            type=MetricType.TRADE_PNL,
            value=pnl,
            tags={'symbol': symbol}
        )
        
        # 滑点
        self.record_metric(
            name=f"trade_slippage_{symbol}",
            type=MetricType.TRADE_SLIPPAGE,
            value=slippage,
            tags={'symbol': symbol}
        )
    
    def register_component(self, component_name: str, monitor_func: Optional[Callable] = None):
        """注册组件监控"""
        self.registered_components.add(component_name)
        
        if monitor_func:
            self.component_monitors[component_name] = monitor_func
            
        self.logger.info(f"组件 {component_name} 已注册监控")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def add_health_check_callback(self, callback: Callable[[HealthCheck], None]):
        """添加健康检查回调"""
        self.health_check_callbacks.append(callback)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {
            'system': {
                'cpu_usage': self.current_metrics.get('system_cpu:cpu_usage', 0),
                'memory_usage': self.current_metrics.get('system_memory:memory_usage', 0),
                'disk_usage': self.current_metrics.get('system_disk:disk_usage', 0)
            },
            'application': {
                'error_rate': self.current_metrics.get('app_error_rate:error_rate', 0),
                'active_alerts': len(self.active_alerts),
                'components': len(self.registered_components)
            },
            'performance': {}
        }
        
        # 添加延迟统计
        for operation, latencies in self.performance_stats['latency_percentiles'].items():
            if latencies:
                latency_array = np.array(list(latencies))
                summary['performance'][operation] = {
                    'p50': np.percentile(latency_array, 50),
                    'p95': np.percentile(latency_array, 95),
                    'p99': np.percentile(latency_array, 99),
                    'mean': np.mean(latency_array)
                }
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        # 计算整体健康状态
        all_statuses = [check.status for check in self.health_checks.values()]
        
        if any(status == HealthStatus.CRITICAL for status in all_statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.UNHEALTHY for status in all_statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in all_statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'overall_status': overall_status.value,
            'checks': {
                name: {
                    'status': check.status.value,
                    'message': check.message,
                    'details': check.details
                }
                for name, check in self.health_checks.items()
            },
            'timestamp': TimeUtils.now_timestamp()
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        return {
            'timestamp': TimeUtils.now_timestamp(),
            'metrics_summary': self.get_metrics_summary(),
            'health_status': self.get_health_status(),
            'active_alerts': self.get_active_alerts(),
            'statistics': {
                'total_metrics_collected': sum(len(buffer) for buffer in self.metrics_buffer.values()),
                'alert_history_count': len(self.alert_history),
                'registered_components': list(self.registered_components)
            }
        }