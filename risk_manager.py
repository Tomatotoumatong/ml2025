# risk_manager.py - 风险管理模块
# =============================================================================
# 核心职责：
# 1. 仓位控制和资金管理
# 2. 止损止盈策略执行
# 3. 最大回撤监控和限制
# 4. 风险敞口计算和控制
# 5. 与ML/RL模型协同的动态风控
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path
from datetime import datetime, timedelta

from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils
from database_manager import DatabaseManager
from market_environment import MarketEnvironmentClassifier, MarketState
from ml_strategy import TradingAction


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    holding_time: int  # 分钟
    max_price: float  # 持仓期间最高价
    min_price: float  # 持仓期间最低价
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    @property
    def market_value(self) -> float:
        """计算市值"""
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        """计算盈亏百分比"""
        if self.avg_price == 0:
            return 0
        return (self.current_price - self.avg_price) / self.avg_price * 100
    
    def update_price(self, new_price: float):
        """更新价格和相关指标"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.avg_price) * self.quantity
        self.max_price = max(self.max_price, new_price)
        self.min_price = min(self.min_price, new_price)
        
        # 更新移动止损
        if self.trailing_stop_distance and self.quantity > 0:
            new_stop = new_price - self.trailing_stop_distance
            if self.stop_loss is None or new_stop > self.stop_loss:
                self.stop_loss = new_stop


@dataclass
class RiskMetrics:
    """风险指标"""
    total_exposure: float  # 总风险敞口
    position_concentration: float  # 仓位集中度
    current_drawdown: float  # 当前回撤
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    leverage: float  # 杠杆率
    risk_score: float  # 综合风险评分 (0-100)
    risk_level: RiskLevel  # 风险等级


class RiskManager:
    """
    风险管理器
    
    核心功能：
    1. 实时风险监控和评估
    2. 动态仓位管理
    3. 止损止盈执行
    4. 与市场环境联动的风控策略
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("RISK_MANAGER")
        
        # 初始化组件
        self.db_manager = DatabaseManager(config_path)
        self.market_classifier = MarketEnvironmentClassifier()
        
        # 风控参数配置
        self.max_position_size = self.config.get("risk.max_position_size", 0.1)
        self.max_total_exposure = self.config.get("risk.max_total_exposure", 0.8)
        self.max_drawdown_limit = self.config.get("risk.max_drawdown_limit", 0.15)
        self.stop_loss_percentage = self.config.get("risk.stop_loss_percentage", 0.02)
        self.take_profit_percentage = self.config.get("risk.take_profit_percentage", 0.04)
        self.trailing_stop_percentage = self.config.get("risk.trailing_stop_percentage", 0.015)
        
        # 市场状态特定的风控参数
        self.market_state_adjustments = self._init_market_state_adjustments()
        
        # 账户和持仓信息
        self.account_balance = self.config.get("account.initial_balance", 100000)
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # 风险指标缓存
        self.risk_metrics_history = deque(maxlen=1000)
        self.daily_returns = deque(maxlen=252)  # 一年交易日
        self.peak_balance = self.account_balance
        
        # 风控状态
        self.is_risk_breached = False
        self.risk_breach_reasons = []
        self.last_risk_check = TimeUtils.now_timestamp()
        
        # 动态风控参数
        self.dynamic_stop_loss_enabled = self.config.get("risk.dynamic_stop_loss", True)
        self.volatility_adjusted_sizing = self.config.get("risk.volatility_adjusted_sizing", True)
        
        self.logger.info("风险管理器初始化完成")
    
    def _init_market_state_adjustments(self) -> Dict[str, Dict[str, float]]:
        """初始化市场状态特定的风控调整参数"""
        return {
            MarketState.BULL_TREND.value: {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.2,
                'max_exposure_multiplier': 1.1
            },
            MarketState.BEAR_TREND.value: {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.9,
                'take_profit_multiplier': 0.9,
                'max_exposure_multiplier': 0.8
            },
            MarketState.HIGH_VOLATILITY.value: {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.5,
                'max_exposure_multiplier': 0.6
            },
            MarketState.CRISIS.value: {
                'position_size_multiplier': 0.3,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 0.8,
                'max_exposure_multiplier': 0.4
            },
            MarketState.LOW_VOLATILITY.value: {
                'position_size_multiplier': 1.3,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.8,
                'max_exposure_multiplier': 1.2
            }
        }
    
    def check_trade_risk(self, 
                        action: TradingAction,
                        symbol: str,
                        quantity: float,
                        current_price: float,
                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查交易风险
        
        Args:
            action: 交易动作
            symbol: 交易标的
            quantity: 交易数量
            current_price: 当前价格
            market_data: 市场数据
        
        Returns:
            风险检查结果
        """
        try:
            # 获取当前市场状态
            market_state, confidence = self.market_classifier.classify_market_environment(market_data)
            
            # 计算交易价值
            trade_value = quantity * current_price
            
            # 风险检查结果
            risk_check = {
                'approved': True,
                'reason': '',
                'risk_level': RiskLevel.LOW,
                'suggested_quantity': quantity,
                'stop_loss': None,
                'take_profit': None,
                'risk_metrics': {}
            }
            
            # 1. 检查总体风险状态
            if self.is_risk_breached:
                risk_check['approved'] = False
                risk_check['reason'] = f"系统风险已触发: {', '.join(self.risk_breach_reasons)}"
                return risk_check
            
            # 2. 仓位大小检查
            position_check = self._check_position_size(symbol, quantity, current_price, market_state)
            if not position_check['approved']:
                risk_check.update(position_check)
                return risk_check
            
            # 3. 总暴露度检查
            exposure_check = self._check_total_exposure(trade_value, market_state)
            if not exposure_check['approved']:
                risk_check.update(exposure_check)
                return risk_check
            
            # 4. 市场状态风险检查
            market_risk_check = self._check_market_state_risk(action, market_state, confidence)
            if not market_risk_check['approved']:
                risk_check.update(market_risk_check)
                return risk_check
            
            # 5. 波动率调整
            if self.volatility_adjusted_sizing:
                adjusted_quantity = self._adjust_quantity_by_volatility(
                    quantity, market_data, market_state
                )
                risk_check['suggested_quantity'] = adjusted_quantity
            
            # 6. 计算止损止盈位
            stops = self._calculate_stop_levels(
                action, current_price, market_data, market_state
            )
            risk_check.update(stops)
            
            # 7. 计算风险指标
            risk_metrics = self._calculate_position_risk_metrics(
                symbol, quantity, current_price, stops['stop_loss']
            )
            risk_check['risk_metrics'] = risk_metrics
            risk_check['risk_level'] = self._assess_risk_level(risk_metrics)
            
            return risk_check
            
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return {
                'approved': False,
                'reason': f"风险检查异常: {str(e)}",
                'risk_level': RiskLevel.HIGH
            }
    
    def _check_position_size(self, symbol: str, quantity: float, 
                           current_price: float, market_state: MarketState) -> Dict[str, Any]:
        """检查仓位大小限制"""
        # 获取市场状态调整系数
        adjustments = self.market_state_adjustments.get(market_state.value, {})
        size_multiplier = adjustments.get('position_size_multiplier', 1.0)
        
        # 计算调整后的最大仓位
        adjusted_max_position = self.max_position_size * size_multiplier
        
        # 计算仓位占比
        position_value = quantity * current_price
        position_ratio = position_value / self.account_balance
        
        if position_ratio > adjusted_max_position:
            return {
                'approved': False,
                'reason': f"仓位过大: {position_ratio:.2%} > {adjusted_max_position:.2%}",
                'suggested_quantity': (adjusted_max_position * self.account_balance) / current_price
            }
        
        return {'approved': True}
    
    def _check_total_exposure(self, new_trade_value: float, 
                            market_state: MarketState) -> Dict[str, Any]:
        """检查总暴露度"""
        # 计算当前总暴露
        current_exposure = sum(pos.market_value for pos in self.positions.values())
        
        # 获取市场状态调整
        adjustments = self.market_state_adjustments.get(market_state.value, {})
        exposure_multiplier = adjustments.get('max_exposure_multiplier', 1.0)
        
        # 计算调整后的最大暴露
        adjusted_max_exposure = self.max_total_exposure * exposure_multiplier
        
        # 预计总暴露
        projected_exposure = (current_exposure + new_trade_value) / self.account_balance
        
        if projected_exposure > adjusted_max_exposure:
            return {
                'approved': False,
                'reason': f"总暴露度过高: {projected_exposure:.2%} > {adjusted_max_exposure:.2%}"
            }
        
        return {'approved': True}
    
    def _check_market_state_risk(self, action: TradingAction, 
                               market_state: MarketState,
                               confidence: float) -> Dict[str, Any]:
        """检查市场状态相关风险"""
        # 危机模式特殊处理
        if market_state == MarketState.CRISIS and confidence > 0.8:
            if action == TradingAction.BUY:
                return {
                    'approved': False,
                    'reason': "危机模式下限制买入",
                    'risk_level': RiskLevel.CRITICAL
                }
        
        # 高波动市场限制
        if market_state == MarketState.HIGH_VOLATILITY:
            if self.get_current_risk_metrics().leverage > 1.0:
                return {
                    'approved': False,
                    'reason': "高波动市场禁止杠杆交易",
                    'risk_level': RiskLevel.HIGH
                }
        
        return {'approved': True}
    
    def _adjust_quantity_by_volatility(self, quantity: float, 
                                     market_data: pd.DataFrame,
                                     market_state: MarketState) -> float:
        """根据波动率调整仓位大小"""
        try:
            # 计算历史波动率
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 20:
                return quantity
            
            # 年化波动率
            volatility = returns.iloc[-20:].std() * np.sqrt(252)
            
            # 目标波动率（根据市场状态调整）
            if market_state in [MarketState.CRISIS, MarketState.HIGH_VOLATILITY]:
                target_volatility = 0.10  # 10%
            elif market_state == MarketState.LOW_VOLATILITY:
                target_volatility = 0.20  # 20%
            else:
                target_volatility = 0.15  # 15%
            
            # 波动率调整系数
            vol_adjustment = min(target_volatility / volatility, 1.5)
            
            return quantity * vol_adjustment
            
        except Exception as e:
            self.logger.error(f"波动率调整失败: {e}")
            return quantity
    
    def _calculate_stop_levels(self, action: TradingAction, current_price: float,
                             market_data: pd.DataFrame, 
                             market_state: MarketState) -> Dict[str, Optional[float]]:
        """计算止损止盈位"""
        # 获取市场状态调整
        adjustments = self.market_state_adjustments.get(market_state.value, {})
        sl_multiplier = adjustments.get('stop_loss_multiplier', 1.0)
        tp_multiplier = adjustments.get('take_profit_multiplier', 1.0)
        
        # 基础止损止盈百分比
        sl_pct = self.stop_loss_percentage * sl_multiplier
        tp_pct = self.take_profit_percentage * tp_multiplier
        
        # 动态止损调整
        if self.dynamic_stop_loss_enabled:
            # 基于ATR的动态止损
            if 'atr' in market_data.columns and not market_data['atr'].empty:
                atr = market_data['atr'].iloc[-1]
                atr_multiplier = 2.0  # ATR倍数
                
                if action == TradingAction.BUY:
                    dynamic_sl = current_price - (atr * atr_multiplier)
                    static_sl = current_price * (1 - sl_pct)
                    stop_loss = max(dynamic_sl, static_sl)  # 取较近的止损
                    
                    take_profit = current_price * (1 + tp_pct)
                    
                elif action == TradingAction.SELL:
                    dynamic_sl = current_price + (atr * atr_multiplier)
                    static_sl = current_price * (1 + sl_pct)
                    stop_loss = min(dynamic_sl, static_sl)
                    
                    take_profit = current_price * (1 - tp_pct)
                else:
                    return {'stop_loss': None, 'take_profit': None}
            else:
                # 使用静态止损
                if action == TradingAction.BUY:
                    stop_loss = current_price * (1 - sl_pct)
                    take_profit = current_price * (1 + tp_pct)
                elif action == TradingAction.SELL:
                    stop_loss = current_price * (1 + sl_pct)
                    take_profit = current_price * (1 - tp_pct)
                else:
                    return {'stop_loss': None, 'take_profit': None}
        else:
            # 静态止损止盈
            if action == TradingAction.BUY:
                stop_loss = current_price * (1 - sl_pct)
                take_profit = current_price * (1 + tp_pct)
            elif action == TradingAction.SELL:
                stop_loss = current_price * (1 + sl_pct)
                take_profit = current_price * (1 - tp_pct)
            else:
                return {'stop_loss': None, 'take_profit': None}
        
        # 添加移动止损距离
        trailing_stop_distance = current_price * self.trailing_stop_percentage
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop_distance': trailing_stop_distance
        }
    
    def _calculate_position_risk_metrics(self, symbol: str, quantity: float,
                                       current_price: float, 
                                       stop_loss: Optional[float]) -> Dict[str, float]:
        """计算持仓风险指标"""
        position_value = quantity * current_price
        
        # 计算风险金额
        if stop_loss:
            risk_per_share = abs(current_price - stop_loss)
            risk_amount = risk_per_share * quantity
        else:
            risk_amount = position_value * self.stop_loss_percentage
        
        # 风险占比
        risk_percentage = risk_amount / self.account_balance
        
        # 风险回报比
        if stop_loss and self.take_profit_percentage > 0:
            reward = position_value * self.take_profit_percentage
            risk_reward_ratio = reward / risk_amount if risk_amount > 0 else 0
        else:
            risk_reward_ratio = 2.0  # 默认风险回报比
        
        # Kelly准则仓位建议
        win_rate = 0.55  # 假设胜率，实际应从历史数据计算
        kelly_fraction = (win_rate * risk_reward_ratio - (1 - win_rate)) / risk_reward_ratio
        kelly_position = max(0, min(kelly_fraction, 0.25))  # 限制最大25%
        
        return {
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'risk_reward_ratio': risk_reward_ratio,
            'kelly_position': kelly_position,
            'position_ratio': position_value / self.account_balance
        }
    
    def _assess_risk_level(self, risk_metrics: Dict[str, float]) -> RiskLevel:
        """评估风险等级"""
        risk_percentage = risk_metrics.get('risk_percentage', 0)
        position_ratio = risk_metrics.get('position_ratio', 0)
        
        # 综合评分
        risk_score = (risk_percentage * 100 + position_ratio * 50) / 1.5
        
        if risk_score < 20:
            return RiskLevel.LOW
        elif risk_score < 40:
            return RiskLevel.MEDIUM
        elif risk_score < 60:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def update_position(self, symbol: str, quantity: float, price: float,
                       action: TradingAction, stops: Dict[str, Optional[float]]):
        """更新持仓信息"""
        if symbol not in self.positions:
            # 新建持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                holding_time=0,
                max_price=price,
                min_price=price,
                stop_loss=stops.get('stop_loss'),
                take_profit=stops.get('take_profit'),
                trailing_stop_distance=stops.get('trailing_stop_distance')
            )
        else:
            # 更新现有持仓
            pos = self.positions[symbol]
            
            if action == TradingAction.BUY:
                # 加仓
                new_quantity = pos.quantity + quantity
                pos.avg_price = (pos.quantity * pos.avg_price + quantity * price) / new_quantity
                pos.quantity = new_quantity
            elif action == TradingAction.SELL:
                # 减仓
                if quantity >= pos.quantity:
                    # 平仓
                    pos.realized_pnl = (price - pos.avg_price) * pos.quantity
                    self.closed_positions.append(pos)
                    del self.positions[symbol]
                    return
                else:
                    # 部分平仓
                    pos.realized_pnl += (price - pos.avg_price) * quantity
                    pos.quantity -= quantity
            
            # 更新止损止盈
            if stops.get('stop_loss'):
                pos.stop_loss = stops['stop_loss']
            if stops.get('take_profit'):
                pos.take_profit = stops['take_profit']
            if stops.get('trailing_stop_distance'):
                pos.trailing_stop_distance = stops['trailing_stop_distance']
    
    def check_stop_conditions(self, market_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """检查止损止盈条件"""
        triggered_stops = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in market_prices:
                continue
            
            current_price = market_prices[symbol]
            position.update_price(current_price)
            
            # 检查止损
            if position.stop_loss:
                if (position.quantity > 0 and current_price <= position.stop_loss) or \
                   (position.quantity < 0 and current_price >= position.stop_loss):
                    triggered_stops.append({
                        'symbol': symbol,
                        'type': OrderType.STOP_LOSS,
                        'quantity': abs(position.quantity),
                        'trigger_price': position.stop_loss,
                        'reason': '触发止损'
                    })
                    self.logger.warning(f"{symbol} 触发止损 @ {position.stop_loss:.2f}")
            
            # 检查止盈
            if position.take_profit:
                if (position.quantity > 0 and current_price >= position.take_profit) or \
                   (position.quantity < 0 and current_price <= position.take_profit):
                    triggered_stops.append({
                        'symbol': symbol,
                        'type': OrderType.TAKE_PROFIT,
                        'quantity': abs(position.quantity),
                        'trigger_price': position.take_profit,
                        'reason': '触发止盈'
                    })
                    self.logger.info(f"{symbol} 触发止盈 @ {position.take_profit:.2f}")
            
            # 更新持仓时间
            position.holding_time += 1
        
        return triggered_stops
    
    def get_current_risk_metrics(self) -> RiskMetrics:
        """获取当前风险指标"""
        # 计算总暴露
        total_exposure = sum(pos.market_value for pos in self.positions.values())
        exposure_ratio = total_exposure / self.account_balance if self.account_balance > 0 else 0
        
        # 计算仓位集中度
        if self.positions:
            position_values = [pos.market_value for pos in self.positions.values()]
            max_position = max(position_values)
            position_concentration = max_position / total_exposure if total_exposure > 0 else 0
        else:
            position_concentration = 0
        
        # 计算当前回撤
        current_balance = self.calculate_total_equity()
        current_drawdown = (current_balance - self.peak_balance) / self.peak_balance if self.peak_balance > 0 else 0
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # 计算最大回撤
        max_drawdown = min(self.risk_metrics_history[-1].max_drawdown if self.risk_metrics_history else 0, 
                           current_drawdown)
        
        # 计算夏普比率
        if len(self.daily_returns) >= 20:
            returns = np.array(list(self.daily_returns))
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 计算VaR和CVaR
        if len(self.daily_returns) >= 20:
            returns = np.array(list(self.daily_returns))
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        else:
            var_95 = 0
            cvar_95 = 0
        
        # 计算杠杆率
        leverage = total_exposure / self.account_balance if self.account_balance > 0 else 0
        
        # 计算综合风险评分
        risk_score = self._calculate_risk_score(
            exposure_ratio, position_concentration, current_drawdown, leverage
        )
        
        # 评估风险等级
        if risk_score < 30:
            risk_level = RiskLevel.LOW
        elif risk_score < 60:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 80:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        metrics = RiskMetrics(
            total_exposure=total_exposure,
            position_concentration=position_concentration,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            leverage=leverage,
            risk_score=risk_score,
            risk_level=risk_level
        )
        
        # 记录历史
        self.risk_metrics_history.append(metrics)
        
        # 检查风险限制
        self._check_risk_limits(metrics)
        
        return metrics
    
    def _calculate_risk_score(self, exposure_ratio: float, concentration: float,
                            drawdown: float, leverage: float) -> float:
        """计算综合风险评分"""
        # 各项权重
        weights = {
            'exposure': 0.3,
            'concentration': 0.2,
            'drawdown': 0.3,
            'leverage': 0.2
        }
        
        # 标准化各项指标到0-100
        exposure_score = min(exposure_ratio * 100, 100)
        concentration_score = concentration * 100
        drawdown_score = abs(drawdown) * 200  # 50%回撤 = 100分
        leverage_score = min(leverage * 50, 100)  # 2倍杠杆 = 100分
        
        # 加权平均
        risk_score = (
            exposure_score * weights['exposure'] +
            concentration_score * weights['concentration'] +
            drawdown_score * weights['drawdown'] +
            leverage_score * weights['leverage']
        )
        
        return min(risk_score, 100)
    
    def _check_risk_limits(self, metrics: RiskMetrics):
        """检查风险限制"""
        self.risk_breach_reasons = []
        
        # 检查最大回撤
        if abs(metrics.current_drawdown) > self.max_drawdown_limit:
            self.is_risk_breached = True
            self.risk_breach_reasons.append(f"超过最大回撤限制: {metrics.current_drawdown:.2%}")
        
        # 检查总暴露
        if metrics.total_exposure / self.account_balance > self.max_total_exposure:
            self.is_risk_breached = True
            self.risk_breach_reasons.append(f"超过最大暴露限制")
        
        # 检查风险等级
        if metrics.risk_level == RiskLevel.CRITICAL:
            self.is_risk_breached = True
            self.risk_breach_reasons.append("风险等级达到危急")
        
        # 记录风险触发
        if self.is_risk_breached:
            self.logger.error(f"风险限制触发: {', '.join(self.risk_breach_reasons)}")
    
    def calculate_total_equity(self) -> float:
        """计算总权益"""
        # 现金 + 未实现盈亏
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.account_balance + unrealized_pnl
    
    def update_daily_return(self, daily_return: float):
        """更新日收益率"""
        self.daily_returns.append(daily_return)
    
    def reset_risk_breach(self):
        """重置风险触发状态"""
        self.is_risk_breached = False
        self.risk_breach_reasons = []
        self.logger.info("风险触发状态已重置")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """获取持仓摘要"""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'total_pnl': 0,
                'positions': []
            }
        
        positions_info = []
        total_value = 0
        total_pnl = 0
        
        for symbol, pos in self.positions.items():
            pos_info = {
                'symbol': symbol,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'pnl_percentage': pos.pnl_percentage,
                'holding_time': pos.holding_time,
                'risk_level': pos.risk_level.value
            }
            positions_info.append(pos_info)
            total_value += pos.market_value
            total_pnl += pos.unrealized_pnl
        
        return {
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl / self.account_balance * 100 if self.account_balance > 0 else 0,
            'positions': positions_info
        }
    
    def save_state(self, filepath: str):
        """保存风控状态"""
        state = {
            'account_balance': self.account_balance,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'holding_time': pos.holding_time,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for symbol, pos in self.positions.items()
            },
            'peak_balance': self.peak_balance,
            'is_risk_breached': self.is_risk_breached,
            'risk_breach_reasons': self.risk_breach_reasons,
            'daily_returns': list(self.daily_returns),
            'timestamp': TimeUtils.now_timestamp()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"风控状态已保存: {filepath}")
    
    def load_state(self, filepath: str):
        """加载风控状态"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.account_balance = state['account_balance']
            self.peak_balance = state['peak_balance']
            self.is_risk_breached = state['is_risk_breached']
            self.risk_breach_reasons = state['risk_breach_reasons']
            self.daily_returns = deque(state['daily_returns'], maxlen=252)
            
            # 恢复持仓
            self.positions.clear()
            for symbol, pos_data in state['positions'].items():
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=pos_data['quantity'],
                    avg_price=pos_data['avg_price'],
                    current_price=pos_data['current_price'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    realized_pnl=pos_data['realized_pnl'],
                    holding_time=pos_data['holding_time'],
                    max_price=pos_data['current_price'],
                    min_price=pos_data['current_price'],
                    stop_loss=pos_data.get('stop_loss'),
                    take_profit=pos_data.get('take_profit')
                )
            
            self.logger.info(f"风控状态已加载: {filepath}")
            
        except Exception as e:
            self.logger.error(f"加载风控状态失败: {e}")