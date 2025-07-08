
# stark4_app.py - 主应用入口
# =============================================================================
# 核心职责：
# 1. 系统初始化和配置加载
# 2. 组件生命周期管理
# 3. 信号流程协调
# 4. 优雅启动和关闭
# =============================================================================

import asyncio
import signal
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from datetime import datetime

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from lock_manager import LockManager
from database_manager import DatabaseManager
from data_collector import DataCollector
from vnpy_integration import VNPYIntegration, TradingSignal
from risk_manager import RiskManager
from meta_model_pipeline import MetaModelPipeline
from system_monitor import SystemMonitor
from fault_tolerance_manager import FaultToleranceManager
from telegram_notifier import TelegramNotifier
from system_state_manager import SystemStateManager


class ApplicationState:
    """应用状态枚举"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class STARK4App:
    """
    STARK4主应用
    
    核心功能：
    1. 组件初始化和依赖注入
    2. 系统启动流程控制
    3. 信号生成和执行协调
    4. 异常处理和恢复
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("STARK4_APP")
        
        # 应用状态
        self.state = ApplicationState.INITIALIZING
        self.start_time = None
        
        # 核心组件
        self.lock_manager: Optional[LockManager] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.data_collector: Optional[DataCollector] = None
        self.vnpy_integration: Optional[VNPYIntegration] = None
        self.risk_manager: Optional[RiskManager] = None
        self.meta_model: Optional[MetaModelPipeline] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.fault_manager: Optional[FaultToleranceManager] = None
        self.notifier: Optional[TelegramNotifier] = None
        self.state_manager: Optional[SystemStateManager] = None
        
        # 运行控制
        self.main_loop_task = None
        self.stop_event = asyncio.Event()
        
        # 性能统计
        self.stats = {
            'signals_generated': 0,
            'orders_executed': 0,
            'orders_success': 0,
            'total_pnl': 0.0
        }
        
        self.logger.info("STARK4应用初始化")
    
    async def initialize(self) -> bool:
        """初始化系统组件"""
        try:
            self.logger.info("开始初始化系统组件...")
            
            # 1. 初始化锁管理器（确保单实例）
            self.lock_manager = LockManager(self.config_path)
            if not self.lock_manager.ensure_single_instance("stark4"):
                self.logger.error("另一个实例正在运行")
                return False
            
            # 2. 初始化状态管理器
            self.state_manager = SystemStateManager(self.config_path)
            
            # 尝试恢复之前的状态
            if await self.state_manager.restore_state():
                self.logger.info("成功恢复系统状态")
            
            # 3. 初始化数据库
            self.db_manager = DatabaseManager(self.config)
            await self.db_manager.initialize()
            
            # 4. 初始化系统监控
            self.system_monitor = SystemMonitor(self.config_path)
            await self.system_monitor.start()
            
            # 5. 初始化故障管理器
            self.fault_manager = FaultToleranceManager(self.config_path)
            await self.fault_manager.start()
            
            # 6. 初始化通知器
            if self.config.get("notifications.telegram_enabled", False):
                self.notifier = TelegramNotifier(self.config_path)
                await self.notifier.initialize()
            
            # 7. 初始化数据收集器
            self.data_collector = DataCollector(self.config_path)
            await self.data_collector.initialize()
            
            # 8. 初始化VNPy集成
            self.vnpy_integration = VNPYIntegration(self.config_path)
            gateway_settings = self.config.get("vnpy.gateway_settings", {})
            if not self.vnpy_integration.connect(gateway_settings):
                raise Exception("VNPy连接失败")
            
            # 订阅行情
            symbols = self.config.get("trading.symbols", [])
            self.vnpy_integration.subscribe(symbols)
            
            # 9. 初始化风控管理器
            self.risk_manager = RiskManager(self.config_path)
            await self.risk_manager.initialize()
            
            # 10. 初始化元模型
            self.meta_model = MetaModelPipeline(self.config_path)
            await self.meta_model.initialize()
            
            # 注册组件到监控器
            self._register_components()
            
            # 注册信号处理
            self._setup_signal_handlers()
            
            self.state = ApplicationState.RUNNING
            self.start_time = datetime.now()
            
            self.logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            self.state = ApplicationState.ERROR
            await self.cleanup()
            return False
    
    def _register_components(self):
        """注册组件到监控器"""
        components = {
            'data_collector': self.data_collector,
            'vnpy_integration': self.vnpy_integration,
            'risk_manager': self.risk_manager,
            'meta_model': self.meta_model
        }
        
        for name, component in components.items():
            if component and self.system_monitor:
                self.system_monitor.register_component(
                    name, 
                    lambda: self._check_component_health(component)
                )
    
    def _check_component_health(self, component) -> Dict[str, Any]:
        """检查组件健康状态"""
        # 这里应该调用各组件的健康检查方法
        # 暂时返回健康状态
        return {'healthy': True}
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理系统信号"""
        self.logger.info(f"收到信号 {signum}，准备关闭...")
        asyncio.create_task(self.shutdown())
    
    async def run(self):
        """运行主循环"""
        self.logger.info("启动主循环")
        
        try:
            # 启动主交易循环
            self.main_loop_task = asyncio.create_task(self._main_trading_loop())
            
            # 等待停止信号
            await self.stop_event.wait()
            
        except Exception as e:
            self.logger.error(f"主循环异常: {e}")
            self.state = ApplicationState.ERROR
        finally:
            await self.shutdown()
    
    async def _main_trading_loop(self):
        """主交易循环"""
        self.logger.info("交易循环已启动")
        
        # 获取交易品种
        symbols = self.config.get("trading.symbols", [])
        check_interval = self.config.get("trading.check_interval", 1)  # 秒
        
        while self.state == ApplicationState.RUNNING:
            try:
                # 生成交易信号
                for symbol in symbols:
                    signal = await self._generate_signal(symbol)
                    
                    if signal and signal.action != "hold":
                        # 执行交易
                        await self._execute_signal(signal)
                
                # 定期保存状态
                if self.stats['signals_generated'] % 100 == 0:
                    await self.state_manager.save_state({
                        'stats': self.stats,
                        'positions': self.vnpy_integration.get_positions()
                    })
                
                # 等待下一个检查周期
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"交易循环异常: {e}")
                
                # 通知故障管理器
                if self.fault_manager:
                    await self.fault_manager.report_fault(
                        "main_loop",
                        str(e),
                        {"component": "trading_loop"}
                    )
    
    async def _generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """生成交易信号"""
        try:
            # 获取最新市场数据
            market_data = await self.data_collector.get_latest_data(symbol)
            
            if not market_data:
                return None
            
            # 使用元模型生成信号
            signal_data = await self.meta_model.predict(market_data)
            
            if signal_data:
                signal = TradingSignal(
                    symbol=symbol,
                    action=signal_data['action'],
                    size=signal_data['size'],
                    confidence=signal_data['confidence'],
                    model_type=signal_data['model_type'],
                    metadata=signal_data.get('metadata', {})
                )
                
                self.stats['signals_generated'] += 1
                
                # 记录信号
                self.logger.info(
                    f"生成信号: {symbol} {signal.action} "
                    f"{signal.size} (置信度: {signal.confidence:.2f})"
                )
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"生成信号失败 {symbol}: {e}")
            return None
    
    async def _execute_signal(self, signal: TradingSignal):
        """执行交易信号"""
        try:
            # 执行交易
            result = await self.vnpy_integration.execute_signal(signal)
            
            self.stats['orders_executed'] += 1
            
            if result.status in ['filled', 'partial_filled']:
                self.stats['orders_success'] += 1
                
                # 计算盈亏（这里需要更复杂的计算）
                # self.stats['total_pnl'] += result.pnl
            
            # 发送通知
            if self.notifier:
                await self.notifier.send_trade_notification(result)
            
            # 记录到监控系统
            if self.system_monitor:
                self.system_monitor.record_trade_result(result)
                
        except Exception as e:
            self.logger.error(f"执行信号失败: {e}")
    
    async def pause(self):
        """暂停交易"""
        self.logger.info("暂停交易")
        self.state = ApplicationState.PAUSED
    
    async def resume(self):
        """恢复交易"""
        self.logger.info("恢复交易")
        self.state = ApplicationState.RUNNING
    
    async def shutdown(self):
        """关闭系统"""
        if self.state == ApplicationState.STOPPING:
            return
        
        self.logger.info("开始关闭系统...")
        self.state = ApplicationState.STOPPING
        
        # 取消主循环
        if self.main_loop_task:
            self.main_loop_task.cancel()
        
        # 保存最终状态
        if self.state_manager:
            await self.state_manager.save_state({
                'stats': self.stats,
                'shutdown_time': TimeUtils.now_timestamp()
            })
        
        # 关闭各组件
        await self.cleanup()
        
        self.state = ApplicationState.STOPPED
        self.stop_event.set()
        
        self.logger.info("系统已关闭")
    
    async def cleanup(self):
        """清理资源"""
        components = [
            ('元模型', self.meta_model),
            ('风控管理器', self.risk_manager),
            ('VNPy集成', self.vnpy_integration),
            ('数据收集器', self.data_collector),
            ('通知器', self.notifier),
            ('故障管理器', self.fault_manager),
            ('系统监控', self.system_monitor),
            ('数据库', self.db_manager)
        ]
        
        for name, component in components:
            if component:
                try:
                    if hasattr(component, 'shutdown'):
                        await component.shutdown()
                    elif hasattr(component, 'close'):
                        await component.close()
                    elif hasattr(component, 'disconnect'):
                        component.disconnect()
                    self.logger.info(f"{name}已关闭")
                except Exception as e:
                    self.logger.error(f"关闭{name}失败: {e}")
        
        # 释放进程锁
        if self.lock_manager:
            self.lock_manager.release_all()
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'state': self.state,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': uptime,
            'stats': self.stats,
            'components': {
                'database': bool(self.db_manager),
                'data_collector': bool(self.data_collector),
                'vnpy': bool(self.vnpy_integration),
                'risk_manager': bool(self.risk_manager),
                'meta_model': bool(self.meta_model),
                'monitor': bool(self.system_monitor),
                'notifier': bool(self.notifier)
            }
        }
