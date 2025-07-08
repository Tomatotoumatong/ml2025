# paper_trading_validator.py - 纸面交易验证器
# =============================================================================
# 核心职责：
# 1. 模拟账户管理
# 2. 虚拟订单执行
# 3. 绩效评估
# 4. 风险分析
# 5. 报告生成
# =============================================================================

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils
from vnpy_integration import TradingSignal, OrderResult, OrderStatus


@dataclass
class VirtualAccount:
    """虚拟账户"""
    initial_balance: float
    balance: float
    available: float
    frozen: float = 0.0
    commission_total: float = 0.0
    positions: Dict[str, 'VirtualPosition'] = field(default_factory=dict)
    
    def get_total_value(self, latest_prices: Dict[str, float]) -> float:
        """计算总资产"""
        position_value = sum(
            pos.get_value(latest_prices.get(symbol, pos.avg_price))
            for symbol, pos in self.positions.items()
        )
        return self.balance + position_value


@dataclass
class VirtualPosition:
    """虚拟持仓"""
    symbol: str
    volume: float
    frozen: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    
    def get_value(self, current_price: float) -> float:
        """计算持仓价值"""
        return self.volume * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        return (current_price - self.avg_price) * self.volume


@dataclass
class VirtualOrder:
    """虚拟订单"""
    order_id: str
    signal: TradingSignal
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    created_time: int = field(default_factory=TimeUtils.now_timestamp)
    filled_time: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_return': f"{self.total_return:.2%}",
            'annualized_return': f"{self.annualized_return:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'win_rate': f"{self.win_rate:.2%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'avg_win': f"{self.avg_win:.2f}",
            'avg_loss': f"{self.avg_loss:.2f}",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }


class PaperTradingValidator:
    """
    纸面交易验证器
    
    核心功能：
    1. 模拟真实交易环境
    2. 执行虚拟订单
    3. 计算交易成本
    4. 评估策略绩效
    5. 生成详细报告
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("PAPER_TRADING")
        
        # 账户配置
        self.initial_balance = self.config.get("paper_trading.initial_balance", 1000000)
        self.commission_rate = self.config.get("paper_trading.commission_rate", 0.0003)
        self.slippage_rate = self.config.get("paper_trading.slippage_rate", 0.0001)
        self.interest_rate = self.config.get("paper_trading.interest_rate", 0.03)
        
        # 虚拟账户
        self.account = VirtualAccount(
            initial_balance=self.initial_balance,
            balance=self.initial_balance,
            available=self.initial_balance
        )
        
        # 订单管理
        self.orders: Dict[str, VirtualOrder] = {}
        self.order_history: List[VirtualOrder] = []
        
        # 市场数据
        self.latest_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        
        # 绩效跟踪
        self.equity_curve: List[Tuple[int, float]] = []
        self.trade_records: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
        # 统计信息
        self.start_time = None
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.logger.info(f"纸面交易验证器初始化，初始资金: {self.initial_balance}")
    
    async def start(self):
        """启动验证器"""
        self.start_time = datetime.now()
        self.equity_curve.append((TimeUtils.now_timestamp(), self.initial_balance))
        self.logger.info("纸面交易开始")
    
    async def execute_signal(self, signal: TradingSignal) -> OrderResult:
        """执行交易信号"""
        try:
            # 创建虚拟订单
            order = VirtualOrder(
                order_id=f"PAPER_{TimeUtils.now_timestamp()}",
                signal=signal
            )
            
            self.orders[order.order_id] = order
            
            # 风控检查
            if not self._risk_check(signal):
                order.status = OrderStatus.REJECTED
                return self._create_order_result(order, "风控拒绝")
            
            # 资金检查
            if not self._capital_check(signal):
                order.status = OrderStatus.REJECTED
                return self._create_order_result(order, "资金不足")
            
            # 模拟成交
            await self._simulate_execution(order)
            
            # 更新持仓和账户
            self._update_position(order)
            self._update_account(order)
            
            # 记录交易
            self._record_trade(order)
            
            # 更新净值
            self._update_equity()
            
            return self._create_order_result(order)
            
        except Exception as e:
            self.logger.error(f"执行信号失败: {e}")
            return OrderResult(
                order_id="",
                signal=signal,
                status=OrderStatus.FAILED,
                error_msg=str(e)
            )
    
    def _risk_check(self, signal: TradingSignal) -> bool:
        """风险检查"""
        # 检查持仓限制
        position = self.account.positions.get(signal.symbol)
        current_position = position.volume if position else 0
        
        # 检查最大持仓
        max_position = self.config.get("paper_trading.max_position_size", 100000)
        if abs(current_position + signal.size) > max_position:
            self.logger.warning(f"超过最大持仓限制: {signal.symbol}")
            return False
        
        # 检查持仓集中度
        total_value = self.account.get_total_value(self.latest_prices)
        position_value = abs(signal.size * self.latest_prices.get(signal.symbol, 0))
        concentration = position_value / total_value
        
        max_concentration = self.config.get("paper_trading.max_concentration", 0.3)
        if concentration > max_concentration:
            self.logger.warning(f"持仓集中度过高: {concentration:.2%}")
            return False
        
        return True
    
    def _capital_check(self, signal: TradingSignal) -> bool:
        """资金检查"""
        # 计算所需资金
        price = self.latest_prices.get(signal.symbol, 0)
        if price <= 0:
            return False
        
        required_capital = signal.size * price
        commission = required_capital * self.commission_rate
        
        # 检查可用资金
        if signal.action == "buy":
            total_required = required_capital + commission
            return self.account.available >= total_required
        
        return True
    
    async def _simulate_execution(self, order: VirtualOrder):
        """模拟订单执行"""
        signal = order.signal
        
        # 获取当前价格
        current_price = self.latest_prices.get(signal.symbol, 0)
        if current_price <= 0:
            order.status = OrderStatus.FAILED
            return
        
        # 计算滑点
        if signal.action == "buy":
            slippage = current_price * self.slippage_rate
            execution_price = current_price + slippage
        else:
            slippage = current_price * self.slippage_rate
            execution_price = current_price - slippage
        
        # 模拟部分成交（可选）
        fill_ratio = 1.0  # 假设全部成交
        
        # 更新订单
        order.status = OrderStatus.FILLED
        order.filled_size = signal.size * fill_ratio
        order.avg_price = execution_price
        order.slippage = slippage * order.filled_size
        order.commission = order.filled_size * execution_price * self.commission_rate
        order.filled_time = TimeUtils.now_timestamp()
        
        self.total_commission += order.commission
        self.total_slippage += order.slippage
    
    def _update_position(self, order: VirtualOrder):
        """更新持仓"""
        if order.status != OrderStatus.FILLED:
            return
        
        symbol = order.signal.symbol
        
        if symbol not in self.account.positions:
            self.account.positions[symbol] = VirtualPosition(symbol=symbol)
        
        position = self.account.positions[symbol]
        
        if order.signal.action == "buy":
            # 买入
            new_volume = position.volume + order.filled_size
            if new_volume != 0:
                position.avg_price = (
                    (position.avg_price * position.volume + 
                     order.avg_price * order.filled_size) / new_volume
                )
            position.volume = new_volume
        else:
            # 卖出
            if position.volume >= order.filled_size:
                # 计算实现盈亏
                realized_pnl = (order.avg_price - position.avg_price) * order.filled_size
                position.realized_pnl += realized_pnl
                position.volume -= order.filled_size
            else:
                self.logger.warning(f"卖出数量超过持仓: {symbol}")
        
        position.commission += order.commission
        
        # 清理空仓
        if abs(position.volume) < 1e-6:
            del self.account.positions[symbol]
    
    def _update_account(self, order: VirtualOrder):
        """更新账户"""
        if order.status != OrderStatus.FILLED:
            return
        
        if order.signal.action == "buy":
            # 买入：扣除资金
            cost = order.filled_size * order.avg_price + order.commission
            self.account.balance -= cost
            self.account.available -= cost
        else:
            # 卖出：增加资金
            revenue = order.filled_size * order.avg_price - order.commission
            self.account.balance += revenue
            self.account.available += revenue
        
        self.account.commission_total += order.commission
    
    def _record_trade(self, order: VirtualOrder):
        """记录交易"""
        if order.status != OrderStatus.FILLED:
            return
        
        self.order_history.append(order)
        
        trade_record = {
            'timestamp': order.filled_time,
            'symbol': order.signal.symbol,
            'action': order.signal.action,
            'size': order.filled_size,
            'price': order.avg_price,
            'commission': order.commission,
            'slippage': order.slippage,
            'pnl': 0.0  # 稍后计算
        }
        
        self.trade_records.append(trade_record)
    
    def _update_equity(self):
        """更新净值"""
        total_value = self.account.get_total_value(self.latest_prices)
        self.equity_curve.append((TimeUtils.now_timestamp(), total_value))
        
        # 计算日收益率
        if len(self.equity_curve) >= 2:
            prev_value = self.equity_curve[-2][1]
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def _create_order_result(self, order: VirtualOrder, error_msg: str = "") -> OrderResult:
        """创建订单结果"""
        return OrderResult(
            order_id=order.order_id,
            signal=order.signal,
            status=order.status,
            filled_size=order.filled_size,
            avg_price=order.avg_price,
            commission=order.commission,
            slippage=order.slippage,
            error_msg=error_msg
        )
    
    def update_price(self, symbol: str, price: float):
        """更新价格"""
        self.latest_prices[symbol] = price
        self.price_history[symbol].append((TimeUtils.now_timestamp(), price))
    
    def calculate_performance(self) -> PerformanceMetrics:
        """计算绩效指标"""
        metrics = PerformanceMetrics()
        
        if not self.equity_curve or len(self.equity_curve) < 2:
            return metrics
        
        # 总收益率
        initial_value = self.equity_curve[0][1]
        final_value = self.equity_curve[-1][1]
        metrics.total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率
        if self.start_time:
            days = (datetime.now() - self.start_time).days
            if days > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (365 / days) - 1
        
        # 夏普比率
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            if len(returns_array) > 1:
                avg_return = np.mean(returns_array) * 252  # 年化
                std_return = np.std(returns_array) * np.sqrt(252)  # 年化
                if std_return > 0:
                    metrics.sharpe_ratio = (avg_return - self.interest_rate) / std_return
        
        # 最大回撤
        equity_values = [v for _, v in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        metrics.max_drawdown = max_dd
        
        # 胜率和盈亏比
        winning_trades = []
        losing_trades = []
        
        for trade in self.trade_records:
            if trade['pnl'] > 0:
                winning_trades.append(trade['pnl'])
            elif trade['pnl'] < 0:
                losing_trades.append(abs(trade['pnl']))
        
        metrics.total_trades = len(self.trade_records)
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        if winning_trades:
            metrics.avg_win = np.mean(winning_trades)
        
        if losing_trades:
            metrics.avg_loss = np.mean(losing_trades)
            
        if metrics.avg_loss > 0:
            metrics.profit_factor = (metrics.avg_win * metrics.winning_trades) / (metrics.avg_loss * metrics.losing_trades)
        
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        metrics = self.calculate_performance()
        
        report = {
            'summary': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'initial_balance': self.initial_balance,
                'final_balance': self.account.balance,
                'total_value': self.account.get_total_value(self.latest_prices),
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage
            },
            'performance': metrics.to_dict(),
            'positions': {
                symbol: {
                    'volume': pos.volume,
                    'avg_price': pos.avg_price,
                    'current_price': self.latest_prices.get(symbol, pos.avg_price),
                    'unrealized_pnl': pos.get_unrealized_pnl(
                        self.latest_prices.get(symbol, pos.avg_price)
                    ),
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.account.positions.items()
            },
            'recent_trades': self.trade_records[-10:] if self.trade_records else []
        }
        
        return report
    
    def export_results(self, export_path: Path):
        """导出结果"""
        try:
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 导出交易记录
            if self.trade_records:
                trades_df = pd.DataFrame(self.trade_records)
                trades_df.to_csv(export_path / "trades.csv", index=False)
            
            # 导出净值曲线
            if self.equity_curve:
                equity_df = pd.DataFrame(
                    self.equity_curve, 
                    columns=['timestamp', 'equity']
                )
                equity_df.to_csv(export_path / "equity_curve.csv", index=False)
            
            # 导出报告
            report = self.generate_report()
            with open(export_path / "report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"结果已导出到: {export_path}")
            
        except Exception as e:
            self.logger.error(f"导出结果失败: {e}")
    
    def reset(self):
        """重置验证器"""
        self.account = VirtualAccount(
            initial_balance=self.initial_balance,
            balance=self.initial_balance,
            available=self.initial_balance
        )
        
        self.orders.clear()
        self.order_history.clear()
        self.latest_prices.clear()
        self.price_history.clear()
        self.equity_curve.clear()
        self.trade_records.clear()
        self.daily_returns.clear()
        
        self.start_time = None
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.logger.info("纸面交易验证器已重置")

