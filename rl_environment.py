# rl_environment.py - RL交易环境
# =============================================================================
# 核心职责：
# 1. 提供与ML pipeline深度集成的交易环境
# 2. 支持基于市场状态的动态环境调整
# 3. 实现元模型决策框架下的RL训练
# 4. 管理投资组合状态和风险控制
# =============================================================================

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils
from prepare_rl_data import StateSpace, RewardFunction, TrajectoryBuffer, MarketContext
from ml_strategy import MLStrategy, TradingSignal
from market_environment import MarketEnvironmentClassifier, MarketState
from risk_manager import RiskManager


class TradingAction(Enum):
    """交易动作枚举"""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Portfolio:
    """增强的投资组合状态"""
    # 基础状态
    cash: float
    position: float  # 持仓数量
    avg_price: float  # 平均持仓价格
    total_value: float
    
    # 收益指标
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # 交易统计
    num_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 时间相关
    holding_time: int = 0  # 当前持仓时间（分钟）
    last_trade_time: int = 0
    
    # 风险指标
    max_position_value: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # 性能指标
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    avg_win_loss_ratio: float = 1.0
    
    # 历史记录
    value_history: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_value(self, current_price: float):
        """更新组合价值和相关指标"""
        # 更新价值
        self.total_value = self.cash + self.position * current_price
        self.value_history.append(self.total_value)
        
        # 更新未实现盈亏
        if self.position > 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.position
        else:
            self.unrealized_pnl = 0
        
        # 更新总盈亏
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # 更新最大持仓价值
        position_value = self.position * current_price
        self.max_position_value = max(self.max_position_value, position_value)
        
        # 更新回撤
        if len(self.value_history) > 1:
            peak_value = max(self.value_history)
            self.current_drawdown = (self.total_value - peak_value) / peak_value
            self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
        
        # 更新持仓时间
        if self.position > 0:
            self.holding_time += 1
    
    def execute_trade(self, action: TradingAction, size: float, price: float, 
                     cost: float, timestamp: int):
        """执行交易并更新统计"""
        trade_record = {
            'action': action.name,
            'size': size,
            'price': price,
            'cost': cost,
            'timestamp': timestamp,
            'pnl': 0.0
        }
        
        if action == TradingAction.BUY:
            # 更新持仓和现金
            new_position = self.position + size
            self.avg_price = ((self.position * self.avg_price + size * price) / new_position 
                            if new_position > 0 else price)
            self.position = new_position
            self.cash -= cost
            self.holding_time = 0  # 重置持仓时间
            
        elif action == TradingAction.SELL:
            # 计算实现盈亏
            trade_pnl = (price - self.avg_price) * size
            self.realized_pnl += trade_pnl
            trade_record['pnl'] = trade_pnl
            
            # 更新交易统计
            if trade_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # 更新持仓和现金
            self.position -= size
            if self.position <= 0:
                self.position = 0
                self.avg_price = 0
                self.holding_time = 0
            self.cash += cost  # cost为负数（收入）
        
        # 更新交易记录
        self.num_trades += 1
        self.last_trade_time = timestamp
        self.trade_history.append(trade_record)
        
        # 更新性能指标
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        # 胜率
        total_closed_trades = self.winning_trades + self.losing_trades
        if total_closed_trades > 0:
            self.win_rate = self.winning_trades / total_closed_trades
        
        # 平均盈亏比
        if self.trade_history:
            wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
            losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
            
            if wins and losses:
                self.avg_win_loss_ratio = np.mean(wins) / np.mean(losses)
        
        # 夏普比率（简化计算）
        if len(self.value_history) > 20:
            returns = np.diff(self.value_history[-20:]) / self.value_history[-21:-1]
            if np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def get_state_dict(self) -> Dict[str, float]:
        """获取状态字典用于RL状态构建"""
        return {
            'position_ratio': self.position * self.avg_price / self.total_value if self.total_value > 0 else 0,
            'cash_ratio': self.cash / self.total_value if self.total_value > 0 else 1,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'holding_time': self.holding_time,
            'num_trades_today': self.num_trades,
            'win_rate': self.win_rate,
            'avg_win_loss_ratio': self.avg_win_loss_ratio,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'leverage': (self.position * self.avg_price) / self.cash if self.cash > 0 else 0,
            'concentration_risk': 1.0  # 单一资产，集中度100%
        }


class MetaTradingEnvironment(gym.Env):
    """
    元模型架构下的强化学习交易环境
    
    核心特性：
    1. 与ML pipeline深度集成
    2. 市场状态自适应
    3. 风险管理集成
    4. 支持增量学习
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("META_RL_ENVIRONMENT")
        
        # 环境配置
        self.initial_balance = self.config.get("rl.environment.initial_balance", 10000)
        self.max_position = self.config.get("rl.environment.max_position", 1.0)
        self.min_trade_size = self.config.get("rl.environment.min_trade_size", 0.001)
        self.transaction_cost_rate = self.config.get("rl.environment.transaction_cost", 0.001)
        self.slippage_rate = self.config.get("rl.environment.slippage", 0.0005)
        self.max_steps = self.config.get("rl.environment.max_steps", 1000)
        
        # 初始化组件
        self.state_space = StateSpace(config_path)
        self.reward_function = RewardFunction(config_path)
        self.market_classifier = MarketEnvironmentClassifier()
        self.ml_strategy = MLStrategy(config_path)
        self.risk_manager = RiskManager(config_path)
        
        # 动作空间
        self.action_space = spaces.Discrete(3)
        
        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space.total_dim,),
            dtype=np.float32
        )
        
        # 市场数据
        self.market_data: Optional[pd.DataFrame] = None
        self.ml_selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # 环境状态
        self.current_step = 0
        self.episode_start_idx = 0
        self.portfolio: Optional[Portfolio] = None
        self.market_context: Optional[MarketContext] = None
        self.ml_signal: Optional[TradingSignal] = None
        
        # 性能追踪
        self.episode_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'ml_alignment_rate': 0.0,
            'risk_violations': 0
        }
    
    def set_market_data(self, 
                       market_data: pd.DataFrame,
                       ml_selected_features: List[str],
                       feature_importance: Dict[str, float]):
        """
        设置市场数据和ML特征
        
        这是与ML pipeline集成的关键接口
        """
        self.market_data = market_data
        self.ml_selected_features = ml_selected_features
        self.feature_importance = feature_importance
        
        self.logger.info(f"环境数据设置完成 - 数据长度: {len(market_data)}, "
                        f"ML特征数: {len(ml_selected_features)}")
    
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """重置环境"""
        if self.market_data is None:
            raise ValueError("市场数据未设置")
        
        # 设置起始位置
        if start_idx is not None:
            self.episode_start_idx = start_idx
        else:
            # 随机选择起始点，确保有足够的历史数据
            min_history = 100
            max_start = len(self.market_data) - self.max_steps - 1
            self.episode_start_idx = np.random.randint(min_history, max_start)
        
        self.current_step = 0
        
        # 重置投资组合
        self.portfolio = Portfolio(
            cash=self.initial_balance,
            position=0,
            avg_price=0,
            total_value=self.initial_balance
        )
        
        # 获取初始市场上下文
        self._update_market_context()
        
        # 获取初始ML信号
        self._update_ml_signal()
        
        # 重置性能指标
        self.episode_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'ml_alignment_rate': 0.0,
            'risk_violations': 0
        }
        
        # 获取初始状态
        state = self._get_state()
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        # 获取当前市场数据
        current_idx = self.episode_start_idx + self.current_step
        current_data = self._get_current_market_data()
        
        # 记录执行前的状态
        previous_state = {
            'price': current_data['close'],
            'position': self.portfolio.position,
            'portfolio_value': self.portfolio.total_value
        }
        
        # 风险检查
        risk_check = self.risk_manager.check_trade_risk(
            action=TradingAction(action),
            current_price=current_data['close'],
            portfolio_state=self.portfolio.get_state_dict(),
            market_state=self.market_context.current_state
        )
        
        # 如果风险检查不通过，强制改为持有
        if not risk_check['approved']:
            self.logger.warning(f"风险检查未通过: {risk_check['reason']}")
            action = TradingAction.HOLD.value
            self.episode_metrics['risk_violations'] += 1
        
        # 执行交易
        trade_info = self._execute_action(action, current_data)
        
        # 更新组合价值
        self.portfolio.update_value(current_data['close'])
        
        # 获取当前状态
        current_state = {
            'price': current_data['close'],
            'position': self.portfolio.position,
            'portfolio_value': self.portfolio.total_value,
            'volatility': self._calculate_recent_volatility(),
            'drawdown': self.portfolio.current_drawdown
        }
        
        # 计算奖励
        reward, reward_components = self.reward_function.calculate_reward(
            action=action,
            previous_state=previous_state,
            current_state=current_state,
            market_context=self.market_context,
            ml_signal=self.ml_signal.to_dict() if self.ml_signal else None
        )
        
        # 更新步数
        self.current_step += 1
        
        # 更新市场上下文和ML信号
        self._update_market_context()
        self._update_ml_signal()
        
        # 获取下一状态
        next_state = self._get_state()
        
        # 检查是否结束
        done = self._is_done()
        
        # 构建信息字典
        info = {
            'portfolio': self.portfolio.get_state_dict(),
            'trade_info': trade_info,
            'market_context': {
                'state': self.market_context.current_state.value,
                'confidence': self.market_context.state_confidence,
                'stress_level': self.market_context.stress_level
            },
            'ml_signal': self.ml_signal.to_dict() if self.ml_signal else None,
            'reward_components': reward_components,
            'current_price': current_data['close'],
            'step': self.current_step,
            'risk_check': risk_check
        }
        
        # 更新性能指标
        self._update_episode_metrics(action, reward)
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        # 创建状态向量
        state = self.state_space.create_state_vector(
            market_data=self._get_market_window(),
            portfolio_state=self.portfolio.get_state_dict(),
            ml_selected_features=self.ml_selected_features,
            feature_importance=self.feature_importance
        )
        
        return state
    
    def _get_current_market_data(self) -> Dict[str, float]:
        """获取当前市场数据"""
        current_idx = self.episode_start_idx + self.current_step
        return self.market_data.iloc[current_idx].to_dict()
    
    def _get_market_window(self) -> pd.DataFrame:
        """获取历史数据窗口"""
        current_idx = self.episode_start_idx + self.current_step
        window_size = 100
        start_idx = max(0, current_idx - window_size + 1)
        return self.market_data.iloc[start_idx:current_idx + 1]
    
    def _update_market_context(self):
        """更新市场上下文"""
        market_window = self._get_market_window()
        
        # 分类市场环境
        state, confidence = self.market_classifier.classify_market_environment(market_window)
        
        # 获取市场制度特征
        symbol = market_window.iloc[-1].get('symbol', 'UNKNOWN')
        regime_features = self.market_classifier.get_market_regime_features(symbol)
        
        self.market_context = MarketContext(
            current_state=state,
            state_confidence=confidence,
            regime_stability=regime_features.get('regime_stability', 0.5),
            transition_probability=regime_features.get('transition_probability', {}),
            stress_level=regime_features.get('market_stress_level', 0.0),
            liquidity_score=regime_features.get('liquidity_conditions', 1.0)
        )
    
    def _update_ml_signal(self):
        """更新ML信号"""
        try:
            market_window = self._get_market_window()
            symbol = market_window.iloc[-1].get('symbol', 'UNKNOWN')
            
            # 获取ML信号
            self.ml_signal = self.ml_strategy.generate_signal(market_window, symbol)
            
        except Exception as e:
            self.logger.error(f"获取ML信号失败: {e}")
            self.ml_signal = None
    
    def _execute_action(self, action: int, current_data: Dict[str, float]) -> Dict[str, Any]:
        """执行交易动作"""
        current_price = current_data['close']
        timestamp = current_data.get('timestamp', TimeUtils.now_timestamp())
        
        trade_info = {
            'action': TradingAction(action).name,
            'executed': False,
            'price': current_price,
            'size': 0,
            'cost': 0,
            'slippage': 0
        }
        
        # 计算滑点
        if action == TradingAction.BUY.value:
            slippage = self.slippage_rate
            execution_price = current_price * (1 + slippage)
        elif action == TradingAction.SELL.value:
            slippage = -self.slippage_rate
            execution_price = current_price * (1 + slippage)
        else:
            slippage = 0
            execution_price = current_price
        
        trade_info['slippage'] = slippage
        
        # 执行交易
        if action == TradingAction.BUY.value:
            # 计算买入数量
            available_cash = self.portfolio.cash * 0.95  # 保留5%现金缓冲
            max_buy_size = available_cash / execution_price
            
            # 考虑最大持仓限制
            current_position_value = self.portfolio.position * current_price
            max_allowed_position = self.max_position - self.portfolio.position
            
            buy_size = min(max_buy_size, max_allowed_position)
            
            if buy_size >= self.min_trade_size:
                # 计算交易成本
                trade_value = buy_size * execution_price
                transaction_cost = trade_value * self.transaction_cost_rate
                total_cost = trade_value + transaction_cost
                
                if total_cost <= self.portfolio.cash:
                    # 执行买入
                    self.portfolio.execute_trade(
                        action=TradingAction.BUY,
                        size=buy_size,
                        price=execution_price,
                        cost=total_cost,
                        timestamp=timestamp
                    )
                    
                    trade_info.update({
                        'executed': True,
                        'size': buy_size,
                        'cost': total_cost,
                        'price': execution_price
                    })
        
        elif action == TradingAction.SELL.value:
            # 卖出所有持仓
            if self.portfolio.position >= self.min_trade_size:
                sell_size = self.portfolio.position
                
                # 计算收入（负成本）
                trade_value = sell_size * execution_price
                transaction_cost = trade_value * self.transaction_cost_rate
                net_revenue = trade_value - transaction_cost
                
                # 执行卖出
                self.portfolio.execute_trade(
                    action=TradingAction.SELL,
                    size=sell_size,
                    price=execution_price,
                    cost=-net_revenue,  # 负数表示收入
                    timestamp=timestamp
                )
                
                trade_info.update({
                    'executed': True,
                    'size': sell_size,
                    'cost': -net_revenue,
                    'price': execution_price
                })
        
        return trade_info
    
    def _calculate_recent_volatility(self) -> float:
        """计算近期波动率"""
        try:
            market_window = self._get_market_window()
            returns = market_window['close'].pct_change().dropna()
            
            if len(returns) >= 20:
                return returns.iloc[-20:].std() * np.sqrt(252)
            else:
                return 0.02  # 默认2%年化波动率
                
        except Exception:
            return 0.02
    
    def _is_done(self) -> bool:
        """检查回合是否结束"""
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True
        
        # 资金耗尽（损失90%以上）
        if self.portfolio.total_value < self.initial_balance * 0.1:
            self.logger.warning("资金严重亏损，回合结束")
            return True
        
        # 达到数据末尾
        if self.episode_start_idx + self.current_step >= len(self.market_data) - 1:
            return True
        
        # 风险管理触发停止
        if self.episode_metrics['risk_violations'] > 10:
            self.logger.warning("风险违规次数过多，回合结束")
            return True
        
        return False
    
    def _update_episode_metrics(self, action: int, reward: float):
        """更新回合性能指标"""
        # 更新回报率
        self.episode_metrics['total_return'] = (
            (self.portfolio.total_value - self.initial_balance) / self.initial_balance
        )
        
        # 更新夏普比率
        self.episode_metrics['sharpe_ratio'] = self.portfolio.sharpe_ratio
        
        # 更新最大回撤
        self.episode_metrics['max_drawdown'] = self.portfolio.max_drawdown
        
        # 更新胜率
        self.episode_metrics['win_rate'] = self.portfolio.win_rate
        
        # 更新ML对齐率
        if self.ml_signal and hasattr(self.reward_function, 'ml_signal_alignment'):
            alignment_history = list(self.reward_function.ml_signal_alignment)
            if alignment_history:
                self.episode_metrics['ml_alignment_rate'] = sum(alignment_history) / len(alignment_history)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """获取回合总结"""
        summary = {
            **self.episode_metrics,
            'final_value': self.portfolio.total_value,
            'total_pnl': self.portfolio.total_pnl,
            'num_trades': self.portfolio.num_trades,
            'avg_holding_time': np.mean([t.get('holding_time', 0) for t in self.portfolio.trade_history]) if self.portfolio.trade_history else 0,
            'trade_frequency': self.portfolio.num_trades / max(self.current_step, 1),
            'market_state_distribution': self._get_market_state_distribution()
        }
        
        return summary
    
    def _get_market_state_distribution(self) -> Dict[str, float]:
        """获取市场状态分布"""
        # TODO: 实现市场状态分布统计
        return {}
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            current_data = self._get_current_market_data()
            
            print(f"\n{'='*50}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Market State: {self.market_context.current_state.value} "
                  f"(confidence: {self.market_context.state_confidence:.2f})")
            print(f"Price: ${current_data['close']:.2f}")
            print(f"Portfolio Value: ${self.portfolio.total_value:.2f} "
                  f"({self.episode_metrics['total_return']:+.2%})")
            print(f"Position: {self.portfolio.position:.4f} @ ${self.portfolio.avg_price:.2f}")
            print(f"Cash: ${self.portfolio.cash:.2f}")
            print(f"Unrealized P&L: ${self.portfolio.unrealized_pnl:+.2f}")
            print(f"Realized P&L: ${self.portfolio.realized_pnl:+.2f}")
            print(f"Sharpe Ratio: {self.portfolio.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {self.portfolio.max_drawdown:.2%}")
            
            if self.ml_signal:
                print(f"ML Signal: {self.ml_signal.action.value} "
                      f"(confidence: {self.ml_signal.confidence:.2f})")
            
            print('='*50)
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]