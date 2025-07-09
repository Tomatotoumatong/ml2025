
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from decimal import Decimal
import threading
import queue

from vnpy.event import EventEngine, Event
from vnpy.trader.engine import MainEngine
from vnpy.trader.constant import (
    Exchange, Product, Direction, OrderType, 
    Status, Offset, Interval
)
from vnpy.trader.object import (
    TickData, OrderData, TradeData, PositionData,
    AccountData, SubscribeRequest, ContractData, BarData, OrderRequest
)
from vnpy.trader.utility import round_to
from ml_strategy import TradingSignal, TradingAction
from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils
from risk_manager import RiskManager


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"



@dataclass
class OrderResult:
    """订单执行结果"""
    order_id: str
    signal: TradingSignal
    status: OrderStatus
    filled_size: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: int = field(default_factory=TimeUtils.now_timestamp)
    error_msg: str = ""


class VNPYIntegration:
    """
    VNPy集成接口
    
    核心功能：
    1. 订单生命周期管理
    2. 实时行情处理
    3. 账户状态同步
    4. 风控前置检查
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("VNPY_INTEGRATION")
        
        # VNPy引擎
        self.event_engine = EventEngine()
        self.main_engine = MainEngine(self.event_engine)
        
        # 风控管理器
        self.risk_manager = RiskManager(config_path)
        
        # 订单管理
        self.active_orders: Dict[str, OrderData] = {}
        self.order_results: Dict[str, OrderResult] = {}
        self.signal_queue: queue.Queue = queue.Queue()
        
        # 市场数据缓存
        self.latest_ticks: Dict[str, TickData] = {}
        self.positions: Dict[str, PositionData] = {}
        self.account: Optional[AccountData] = None
        
        # 配置参数
        self.gateway_name = self.config.get("vnpy.gateway", "CTP")
        self.order_timeout = self.config.get("vnpy.order_timeout", 30)
        self.max_retry = self.config.get("vnpy.max_retry", 3)
        self.slippage_tick = self.config.get("vnpy.slippage_tick", 2)
        
        # 回调函数
        self.signal_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        
        # 运行状态
        self.is_running = False
        self.executor_thread = None
        
        self._init_event_handlers()
        self.logger.info("VNPy集成初始化完成")
    
    def _init_event_handlers(self):
        """初始化事件处理器"""
        self.event_engine.register("tick", self._on_tick)
        self.event_engine.register("order", self._on_order)
        self.event_engine.register("trade", self._on_trade)
        self.event_engine.register("position", self._on_position)
        self.event_engine.register("account", self._on_account)
        self.event_engine.register("error", self._on_error)
    
    def connect(self, gateway_setting: Dict[str, str]) -> bool:
        """连接交易网关"""
        try:
            # 添加网关
            gateway_class = self.main_engine.get_gateway(self.gateway_name)
            if not gateway_class:
                self.logger.error(f"网关 {self.gateway_name} 不存在")
                return False
            
            self.main_engine.add_gateway(gateway_class)
            
            # 连接网关
            self.main_engine.connect(gateway_setting, self.gateway_name)
            self.logger.info(f"正在连接网关 {self.gateway_name}")
            
            # 启动执行线程
            self.is_running = True
            self.executor_thread = threading.Thread(
                target=self._signal_executor_loop,
                daemon=True
            )
            self.executor_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"连接网关失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.is_running = False
        
        if self.executor_thread:
            self.executor_thread.join(timeout=5)
        
        self.main_engine.close()
        self.logger.info("已断开网关连接")
    
    def subscribe(self, symbols: List[str], exchange: Exchange = Exchange.SHFE):
        """订阅行情"""
        for symbol in symbols:
            req = SubscribeRequest(
                symbol=symbol,
                exchange=exchange
            )
            self.main_engine.subscribe(req, self.gateway_name)
            self.logger.info(f"订阅行情: {symbol}.{exchange.value}")
    
    async def execute_signal(self, signal: TradingSignal) -> OrderResult:
        """执行交易信号"""
        try:
            # 风控检查
            if not await self._risk_check(signal):
                return OrderResult(
                    order_id="",
                    signal=signal,
                    status=OrderStatus.REJECTED,
                    error_msg="风控检查未通过"
                )
            
            # 添加到执行队列
            self.signal_queue.put(signal)
            
            # 等待执行结果
            timeout = self.order_timeout
            start_time = TimeUtils.now()
            
            while timeout > 0:
                if signal.timestamp in self.order_results:
                    return self.order_results[signal.timestamp]
                
                await asyncio.sleep(0.1)
                timeout -= 0.1
            
            # 超时
            return OrderResult(
                order_id="",
                signal=signal,
                status=OrderStatus.FAILED,
                error_msg="订单执行超时"
            )
            
        except Exception as e:
            self.logger.error(f"执行信号失败: {e}")
            return OrderResult(
                order_id="",
                signal=signal,
                status=OrderStatus.FAILED,
                error_msg=str(e)
            )
    
    def _signal_executor_loop(self):
        """信号执行循环"""
        while self.is_running:
            try:
                # 获取信号（超时1秒）
                signal = self.signal_queue.get(timeout=1)
                
                # 执行交易
                self._execute_order(signal)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"信号执行异常: {e}")
    
    def _execute_order(self, signal: TradingSignal):
        """执行订单"""
        try:
            # 获取最新行情
            tick = self.latest_ticks.get(signal.symbol)
            if not tick:
                self.logger.error(f"无法获取 {signal.symbol} 行情")
                self._record_failed_order(signal, "无行情数据")
                return
            
            # 计算订单价格
            if signal.price:
                order_price = signal.price
            else:
                if signal.action == "buy":
                    order_price = tick.ask_price_1 + self.slippage_tick * tick.pricetick
                else:
                    order_price = tick.bid_price_1 - self.slippage_tick * tick.pricetick
            
            # 创建订单请求
            order_req = OrderRequest(
                symbol=signal.symbol,
                exchange=tick.exchange,
                direction=Direction.LONG if signal.action == "buy" else Direction.SHORT,
                type=OrderType.LIMIT,
                volume=signal.size,
                price=round_to(order_price, tick.pricetick),
                offset=Offset.OPEN
            )
            
            # 发送订单
            order_id = self.main_engine.send_order(order_req, self.gateway_name)
            
            if order_id:
                self.logger.info(
                    f"订单已发送: {signal.symbol} {signal.action} "
                    f"{signal.size}@{order_price}, ID: {order_id}"
                )
                
                # 记录订单映射
                self.active_orders[order_id] = signal
            else:
                self._record_failed_order(signal, "订单发送失败")
                
        except Exception as e:
            self.logger.error(f"执行订单异常: {e}")
            self._record_failed_order(signal, str(e))
    
    def _record_failed_order(self, signal: TradingSignal, error_msg: str):
        """记录失败订单"""
        result = OrderResult(
            order_id="",
            signal=signal,
            status=OrderStatus.FAILED,
            error_msg=error_msg
        )
        self.order_results[signal.timestamp] = result
    
    async def _risk_check(self, signal: TradingSignal) -> bool:
        """风控检查"""
        try:
            # 检查持仓限制
            position = self.positions.get(signal.symbol)
            current_position = position.volume if position else 0
            
            check_result = await self.risk_manager.check_signal(
                symbol=signal.symbol,
                action=signal.action,
                size=signal.size,
                current_position=current_position,
                confidence=signal.confidence
            )
            
            if not check_result['approved']:
                self.logger.warning(
                    f"风控拒绝: {signal.symbol} {signal.action} "
                    f"{signal.size}, 原因: {check_result['reason']}"
                )
                
            return check_result['approved']
            
        except Exception as e:
            self.logger.error(f"风控检查异常: {e}")
            return False
    
    def _on_tick(self, event: Event):
        """处理行情事件"""
        tick: TickData = event.data
        self.latest_ticks[tick.symbol] = tick
    
    def _on_order(self, event: Event):
        """处理订单事件"""
        order: OrderData = event.data
        
        # 更新订单状态
        if order.vt_orderid in self.active_orders:
            signal = self.active_orders[order.vt_orderid]
            
            # 转换订单状态
            if order.status == Status.ALLTRADED:
                status = OrderStatus.FILLED
            elif order.status == Status.PARTTRADED:
                status = OrderStatus.PARTIAL_FILLED
            elif order.status == Status.CANCELLED:
                status = OrderStatus.CANCELLED
            elif order.status == Status.REJECTED:
                status = OrderStatus.REJECTED
            else:
                status = OrderStatus.SUBMITTED
            
            # 创建订单结果
            result = OrderResult(
                order_id=order.vt_orderid,
                signal=signal,
                status=status,
                filled_size=order.traded,
                avg_price=order.price if order.traded > 0 else 0
            )
            
            # 记录结果
            self.order_results[signal.timestamp] = result
            
            # 触发回调
            for callback in self.trade_callbacks:
                callback(result)
            
            # 清理已完成订单
            if order.status in [Status.ALLTRADED, Status.CANCELLED, Status.REJECTED]:
                del self.active_orders[order.vt_orderid]
    
    def _on_trade(self, event: Event):
        """处理成交事件"""
        trade: TradeData = event.data
        self.logger.info(
            f"成交: {trade.symbol} {trade.direction.value} "
            f"{trade.volume}@{trade.price}"
        )
    
    def _on_position(self, event: Event):
        """处理持仓事件"""
        position: PositionData = event.data
        self.positions[position.symbol] = position
    
    def _on_account(self, event: Event):
        """处理账户事件"""
        self.account = event.data
    
    def _on_error(self, event: Event):
        """处理错误事件"""
        error = event.data
        self.logger.error(f"网关错误: {error}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        if not self.account:
            return {}
        
        return {
            'balance': float(self.account.balance),
            'available': float(self.account.available),
            'frozen': float(self.account.frozen),
            'commission': float(self.account.commission),
            'position_profit': float(self.account.position_profit),
            'close_profit': float(self.account.close_profit)
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取持仓信息"""
        positions = {}
        
        for symbol, pos in self.positions.items():
            positions[symbol] = {
                'volume': pos.volume,
                'frozen': pos.frozen,
                'price': float(pos.price),
                'pnl': float(pos.pnl),
                'yd_volume': pos.yd_volume
            }
        
        return positions
    
    def register_signal_callback(self, callback: Callable):
        """注册信号回调"""
        self.signal_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable):
        """注册成交回调"""
        self.trade_callbacks.append(callback)
