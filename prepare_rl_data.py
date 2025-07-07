# prepare_rl_data.py - RL数据准备
# =============================================================================
# 核心职责：
# 1. 接收ML筛选的特征，构建RL状态空间
# 2. 设计适应市场环境的奖励函数
# 3. 管理经验回放缓冲区，支持增量学习
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json
from pathlib import Path
from dataclasses import dataclass
from scipy import stats

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from technical_indicators import TechnicalIndicators
from market_environment import MarketEnvironmentClassifier, MarketState
from ml_pipeline import FeatureEngineering


@dataclass
class MarketContext:
    """市场上下文信息"""
    current_state: MarketState
    state_confidence: float
    regime_stability: float
    transition_probability: Dict[str, float]
    stress_level: float
    liquidity_score: float


class StateSpace:
    """
    状态空间定义 - 与ML特征选择深度集成
    
    关键设计原则：
    1. 动态使用ML选择的特征，而非所有特征
    2. 包含市场环境上下文信息
    3. 适应不同市场状态的特征重要性
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("RL_STATE_SPACE")
        
        # 初始化组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.feature_engineering = FeatureEngineering()
        
        # 状态空间组成部分及其动态维度
        self.state_components = {
            'ml_selected_features': {
                'dim': self.config.get("rl.state.ml_features_dim", 30),
                'weight': 0.5  # ML特征权重最高
            },
            'market_context': {
                'dim': 15,  # 市场环境固定15维
                'weight': 0.3
            },
            'portfolio_state': {
                'dim': 8,
                'weight': 0.1
            },
            'risk_metrics': {
                'dim': 7,
                'weight': 0.1
            }
        }
        
        # 计算总维度
        self.total_dim = sum(comp['dim'] for comp in self.state_components.values())
        
        # ML特征缓存
        self.ml_feature_cache = {}
        self.feature_importance_cache = {}
        
        # 市场状态特征标准化参数
        self.normalization_params = {}
        self._init_normalization_params()
    
    def _init_normalization_params(self):
        """初始化标准化参数"""
        # 为每个市场状态设置不同的标准化参数
        for state in MarketState:
            self.normalization_params[state.value] = {
                'feature_scales': {},
                'initialized': False
            }
    
    def create_state_vector(self, 
                          market_data: pd.DataFrame,
                          portfolio_state: Dict[str, float],
                          ml_selected_features: List[str],
                          feature_importance: Dict[str, float]) -> np.ndarray:
        """
        创建状态向量 - 核心方法
        
        Args:
            market_data: 市场数据
            portfolio_state: 投资组合状态
            ml_selected_features: ML模型选择的特征
            feature_importance: 特征重要性权重
        
        Returns:
            标准化的状态向量
        """
        try:
            # 获取市场上下文
            market_context = self._get_market_context(market_data)
            
            # 缓存特征信息
            self.ml_feature_cache = ml_selected_features
            self.feature_importance_cache = feature_importance
            
            state_vector = []
            
            # 1. ML选择的特征（动态加权）
            ml_features = self._extract_ml_selected_features(
                market_data, ml_selected_features, feature_importance, market_context
            )
            state_vector.extend(ml_features)
            
            # 2. 市场环境特征
            market_features = self._extract_market_context_features(market_context)
            state_vector.extend(market_features)
            
            # 3. 投资组合特征
            portfolio_features = self._extract_portfolio_features(
                portfolio_state, market_context
            )
            state_vector.extend(portfolio_features)
            
            # 4. 风险度量特征
            risk_features = self._extract_risk_features(
                market_data, portfolio_state, market_context
            )
            state_vector.extend(risk_features)
            
            # 转换为numpy数组
            state_vector = np.array(state_vector, dtype=np.float32)
            
            # 确保维度正确
            if len(state_vector) != self.total_dim:
                self.logger.warning(f"状态向量维度不匹配: {len(state_vector)} vs {self.total_dim}")
                state_vector = self._adjust_dimension(state_vector)
            
            # 市场状态自适应标准化
            state_vector = self._adaptive_normalize(state_vector, market_context.current_state)
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"状态向量创建失败: {e}")
            return np.zeros(self.total_dim, dtype=np.float32)
    
    def _get_market_context(self, market_data: pd.DataFrame) -> MarketContext:
        """获取市场上下文信息"""
        # 分类市场环境
        state, confidence = self.market_classifier.classify_market_environment(market_data)
        
        # 获取市场制度特征
        regime_features = self.market_classifier.get_market_regime_features(
            market_data.iloc[-1]['symbol'] if 'symbol' in market_data.columns else "UNKNOWN"
        )
        
        return MarketContext(
            current_state=state,
            state_confidence=confidence,
            regime_stability=regime_features.get('regime_stability', 0.5),
            transition_probability=regime_features.get('transition_probability', {}),
            stress_level=regime_features.get('market_stress_level', 0.0),
            liquidity_score=regime_features.get('liquidity_conditions', 1.0)
        )
    
    def _extract_ml_selected_features(self, 
                                    market_data: pd.DataFrame,
                                    selected_features: List[str],
                                    importance_weights: Dict[str, float],
                                    market_context: MarketContext) -> List[float]:
        """
        提取ML选择的特征并动态加权
        
        这是与ML pipeline深度集成的关键部分
        """
        features = []
        target_dim = self.state_components['ml_selected_features']['dim']
        
        # 根据市场状态调整特征权重
        market_adjustment = self._get_market_state_adjustments(market_context.current_state)
        
        # 按重要性排序特征
        sorted_features = sorted(
            [(f, importance_weights.get(f, 0)) for f in selected_features],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 提取特征值
        for feature_name, importance in sorted_features[:target_dim]:
            if feature_name in market_data.columns:
                value = market_data[feature_name].iloc[-1]
                
                # 处理缺失值
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                
                # 根据市场状态和特征重要性调整权重
                adjusted_importance = importance * market_adjustment.get(feature_name, 1.0)
                weighted_value = value * adjusted_importance
                
                features.append(weighted_value)
            else:
                features.append(0.0)
        
        # 填充到目标维度
        while len(features) < target_dim:
            features.append(0.0)
        
        return features[:target_dim]
    
    def _extract_market_context_features(self, context: MarketContext) -> List[float]:
        """提取市场环境特征"""
        features = []
        
        # 1. 市场状态编码（9维one-hot + confidence）
        state_encoding = [0.0] * 9
        state_index = list(MarketState).index(context.current_state)
        state_encoding[state_index] = context.state_confidence
        features.extend(state_encoding)
        
        # 2. 市场稳定性和压力指标
        features.extend([
            context.regime_stability,
            context.stress_level,
            context.liquidity_score
        ])
        
        # 3. 状态转换概率（前3个最可能的转换）
        top_transitions = sorted(
            context.transition_probability.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for _, prob in top_transitions:
            features.append(prob)
        
        # 填充到15维
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
    
    def _extract_portfolio_features(self, 
                                  portfolio_state: Dict[str, float],
                                  market_context: MarketContext) -> List[float]:
        """提取投资组合特征（考虑市场环境）"""
        features = []
        
        # 基础持仓信息
        position_ratio = portfolio_state.get('position_ratio', 0)
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0)
        holding_time = portfolio_state.get('holding_time', 0)
        
        # 根据市场状态调整特征表示
        if market_context.current_state in [MarketState.HIGH_VOLATILITY, MarketState.CRISIS]:
            # 高波动/危机时，放大风险相关特征
            position_risk = position_ratio * (1 + market_context.stress_level)
            pnl_risk = unrealized_pnl * (1 + market_context.stress_level)
        else:
            position_risk = position_ratio
            pnl_risk = unrealized_pnl
        
        features.extend([
            position_risk,
            pnl_risk,
            np.tanh(holding_time / 1440),  # 持仓时间（天）
            portfolio_state.get('num_trades_today', 0) / 50,  # 标准化交易次数
            portfolio_state.get('win_rate', 0.5),
            portfolio_state.get('avg_win_loss_ratio', 1.0) - 1,  # 盈亏比偏差
            portfolio_state.get('max_drawdown', 0),
            portfolio_state.get('sharpe_ratio', 0)
        ])
        
        return features[:8]
    
    def _extract_risk_features(self, 
                             market_data: pd.DataFrame,
                             portfolio_state: Dict[str, float],
                             market_context: MarketContext) -> List[float]:
        """提取风险特征（市场自适应）"""
        features = []
        
        try:
            returns = market_data['close'].pct_change().dropna()
            
            if len(returns) >= 20:
                # 1. VaR和CVaR
                var_95 = np.percentile(returns[-20:], 5)
                cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                
                # 2. 下行风险
                downside_returns = returns[returns < 0]
                downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
                
                # 3. 尾部风险（峰度）
                kurtosis = stats.kurtosis(returns[-20:])
                
                features.extend([
                    abs(var_95) * (1 + market_context.stress_level),  # 市场压力调整
                    abs(cvar_95) * (1 + market_context.stress_level),
                    downside_std,
                    np.tanh(kurtosis / 3)  # 标准化峰度
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 4. 组合相关风险
            leverage = portfolio_state.get('leverage', 1.0)
            concentration = portfolio_state.get('concentration_risk', 0)
            
            # 根据市场流动性调整风险
            liquidity_adjusted_leverage = leverage * (2 - market_context.liquidity_score)
            
            features.extend([
                liquidity_adjusted_leverage - 1,
                concentration,
                market_context.stress_level  # 直接包含市场压力
            ])
            
        except Exception as e:
            self.logger.error(f"风险特征提取失败: {e}")
            features = [0.0] * 7
        
        return features[:7]
    
    def _adjust_dimension(self, state_vector: np.ndarray) -> np.ndarray:
        """调整状态向量维度"""
        if len(state_vector) < self.total_dim:
            # 填充零
            padding = np.zeros(self.total_dim - len(state_vector))
            state_vector = np.concatenate([state_vector, padding])
        elif len(state_vector) > self.total_dim:
            # 截断
            state_vector = state_vector[:self.total_dim]
        
        return state_vector
    
    def _adaptive_normalize(self, state_vector: np.ndarray, market_state: MarketState) -> np.ndarray:
        """
        市场状态自适应标准化
        
        不同市场状态使用不同的标准化策略
        """
        # 获取当前市场状态的标准化参数
        norm_params = self.normalization_params[market_state.value]
        
        if market_state in [MarketState.HIGH_VOLATILITY, MarketState.CRISIS]:
            # 高波动/危机时使用更保守的标准化
            return np.tanh(state_vector / 3)
        elif market_state in [MarketState.LOW_VOLATILITY, MarketState.SIDEWAYS]:
            # 低波动/横盘时使用更敏感的标准化
            return np.tanh(state_vector * 2)
        else:
            # 其他情况使用标准标准化
            return np.tanh(state_vector)
    
    def _get_market_state_adjustments(self, market_state: MarketState) -> Dict[str, float]:
        """获取不同市场状态下的特征权重调整"""
        adjustments = {
            MarketState.BULL_TREND: {
                'momentum': 1.2, 'rsi': 1.1, 'macd': 1.1,
                'volume': 0.9, 'volatility': 0.8
            },
            MarketState.BEAR_TREND: {
                'support': 1.2, 'oversold': 1.1, 'volatility': 1.1,
                'momentum': 0.8, 'volume': 1.0
            },
            MarketState.HIGH_VOLATILITY: {
                'atr': 1.3, 'bollinger': 1.2, 'volatility': 1.2,
                'momentum': 0.7, 'trend': 0.8
            },
            MarketState.CRISIS: {
                'risk': 1.5, 'support': 1.3, 'liquidity': 1.2,
                'momentum': 0.6, 'trend': 0.7
            }
        }
        
        # 返回通用调整，如果没有特定调整
        default_adjustment = {}
        specific_adjustment = adjustments.get(market_state, {})
        
        # 合并调整
        for feature_type in ['momentum', 'volatility', 'volume', 'trend', 'risk']:
            default_adjustment[feature_type] = specific_adjustment.get(feature_type, 1.0)
        
        return default_adjustment
    """状态空间定义"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("STATE_SPACE")
        
        # 状态空间配置
        self.state_features = self.config.get("rl.state_features", [
            'price_features', 'technical_indicators', 'market_regime', 
            'portfolio_state', 'risk_metrics'
        ])
        
        # 状态维度
        self.price_dim = 10  # 价格相关特征
        self.technical_dim = 20  # 技术指标特征
        self.market_dim = 10  # 市场制度特征
        self.portfolio_dim = 5  # 持仓状态特征
        self.risk_dim = 5  # 风险度量特征
        
        self.total_dim = (self.price_dim + self.technical_dim + 
                         self.market_dim + self.portfolio_dim + self.risk_dim)
        
        # 特征缓冲区
        self.feature_buffer = deque(maxlen=100)
    
    def create_state_vector(self, market_data: pd.DataFrame, 
                          portfolio_state: Dict[str, float],
                          selected_features: List[str] = None) -> np.ndarray:
        """创建状态向量"""
        try:
            state_components = []
            
            # 1. 价格特征
            if 'price_features' in self.state_features:
                price_features = self._extract_price_features(market_data)
                state_components.append(price_features)
            
            # 2. 技术指标特征（使用ML选择的特征）
            if 'technical_indicators' in self.state_features:
                tech_features = self._extract_technical_features(market_data, selected_features)
                state_components.append(tech_features)
            
            # 3. 市场制度特征
            if 'market_regime' in self.state_features:
                market_features = self._extract_market_regime_features(market_data)
                state_components.append(market_features)
            
            # 4. 持仓状态特征
            if 'portfolio_state' in self.state_features:
                portfolio_features = self._extract_portfolio_features(portfolio_state)
                state_components.append(portfolio_features)
            
            # 5. 风险度量特征
            if 'risk_metrics' in self.state_features:
                risk_features = self._extract_risk_features(market_data, portfolio_state)
                state_components.append(risk_features)
            
            # 合并所有特征
            state_vector = np.concatenate(state_components)
            
            # 确保维度正确
            if len(state_vector) != self.total_dim:
                # 填充或截断到正确维度
                if len(state_vector) < self.total_dim:
                    state_vector = np.pad(state_vector, (0, self.total_dim - len(state_vector)))
                else:
                    state_vector = state_vector[:self.total_dim]
            
            # 标准化
            state_vector = self._normalize_state(state_vector)
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"状态向量创建失败: {e}")
            return np.zeros(self.total_dim)
    
    def _extract_price_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """提取价格特征"""
        features = []
        
        try:
            close_prices = market_data['close'].values
            
            # 价格相对位置
            current_price = close_prices[-1]
            ma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else current_price
            ma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else current_price
            
            features.extend([
                (current_price - ma_20) / ma_20,  # 相对MA20
                (current_price - ma_50) / ma_50,  # 相对MA50
                (close_prices[-1] - close_prices[-2]) / close_prices[-2],  # 1期收益率
                (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0,  # 5期收益率
                (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0,  # 10期收益率
            ])
            
            # 价格波动特征
            returns = np.diff(close_prices) / close_prices[:-1]
            features.extend([
                np.std(returns[-20:]) if len(returns) >= 20 else 0,  # 20期波动率
                np.std(returns[-5:]) if len(returns) >= 5 else 0,   # 5期波动率
                np.max(close_prices[-20:]) / np.min(close_prices[-20:]) - 1 if len(close_prices) >= 20 else 0,  # 价格范围
            ])
            
            # 价格动量
            if len(close_prices) >= 20:
                momentum = (close_prices[-1] - close_prices[-20]) / close_prices[-20]
                acceleration = momentum - (close_prices[-10] - close_prices[-20]) / close_prices[-20]
                features.extend([momentum, acceleration])
            else:
                features.extend([0, 0])
            
        except Exception as e:
            self.logger.error(f"价格特征提取失败: {e}")
            features = [0] * self.price_dim
        
        return np.array(features[:self.price_dim])
    
    def _extract_technical_features(self, market_data: pd.DataFrame, 
                                  selected_features: List[str] = None) -> np.ndarray:
        """提取技术指标特征"""
        features = []
        
        try:
            # 如果有ML选择的特征，优先使用
            if selected_features:
                for feature in selected_features[:self.technical_dim]:
                    if feature in market_data.columns:
                        value = market_data[feature].iloc[-1]
                        # 标准化处理
                        if not np.isnan(value) and not np.isinf(value):
                            features.append(value)
                        else:
                            features.append(0)
            
            # 补充必要的技术指标
            if len(features) < self.technical_dim:
                # RSI
                if 'rsi' in market_data.columns:
                    features.append((market_data['rsi'].iloc[-1] - 50) / 50)
                
                # MACD
                if 'macd' in market_data.columns:
                    features.append(market_data['macd'].iloc[-1])
                
                # 布林带位置
                if all(col in market_data.columns for col in ['bb_upper', 'bb_lower', 'close']):
                    close = market_data['close'].iloc[-1]
                    bb_upper = market_data['bb_upper'].iloc[-1]
                    bb_lower = market_data['bb_lower'].iloc[-1]
                    bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                    features.append(bb_position)
                
                # ATR
                if 'atr' in market_data.columns:
                    atr_ratio = market_data['atr'].iloc[-1] / market_data['close'].iloc[-1]
                    features.append(atr_ratio)
            
        except Exception as e:
            self.logger.error(f"技术特征提取失败: {e}")
        
        # 确保长度正确
        features = features[:self.technical_dim]
        if len(features) < self.technical_dim:
            features.extend([0] * (self.technical_dim - len(features)))
        
        return np.array(features)
    
    def _extract_market_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """提取市场制度特征"""
        features = []
        
        try:
            classifier = MarketEnvironmentClassifier()
            
            # 市场状态分类
            state, confidence = classifier.classify_market_environment(market_data)
            
            # One-hot编码市场状态（9种状态）
            state_encoding = [0] * 9
            state_mapping = {
                'bull_trend': 0, 'bear_trend': 1, 'sideways': 2,
                'high_volatility': 3, 'low_volatility': 4, 'crisis': 5,
                'recovery': 6, 'accumulation': 7, 'distribution': 8
            }
            
            if state in state_mapping:
                state_encoding[state_mapping[state]] = confidence
            
            features.extend(state_encoding)
            features.append(confidence)  # 置信度
            
        except Exception as e:
            self.logger.error(f"市场制度特征提取失败: {e}")
            features = [0] * self.market_dim
        
        return np.array(features[:self.market_dim])
    
    def _extract_portfolio_features(self, portfolio_state: Dict[str, float]) -> np.ndarray:
        """提取持仓特征"""
        features = []
        
        try:
            features.extend([
                portfolio_state.get('position_ratio', 0),  # 仓位比例
                portfolio_state.get('unrealized_pnl', 0),  # 未实现盈亏
                portfolio_state.get('holding_time', 0) / 1440,  # 持仓时间（分钟转天）
                portfolio_state.get('num_trades_today', 0) / 10,  # 今日交易次数（标准化）
                portfolio_state.get('win_rate', 0.5)  # 胜率
            ])
            
        except Exception as e:
            self.logger.error(f"持仓特征提取失败: {e}")
            features = [0] * self.portfolio_dim
        
        return np.array(features[:self.portfolio_dim])
    
    def _extract_risk_features(self, market_data: pd.DataFrame, 
                             portfolio_state: Dict[str, float]) -> np.ndarray:
        """提取风险特征"""
        features = []
        
        try:
            # 市场风险指标
            returns = market_data['close'].pct_change().dropna()
            
            # VaR (95%)
            if len(returns) >= 20:
                var_95 = np.percentile(returns[-20:], 5)
                features.append(abs(var_95))
            else:
                features.append(0)
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            features.append(abs(max_drawdown))
            
            # 夏普比率（简化版）
            if len(returns) >= 20:
                sharpe = returns[-20:].mean() / returns[-20:].std() if returns[-20:].std() > 0 else 0
                features.append(sharpe)
            else:
                features.append(0)
            
            # 持仓风险
            features.extend([
                portfolio_state.get('leverage', 1.0) - 1,  # 杠杆率
                portfolio_state.get('concentration_risk', 0)  # 集中度风险
            ])
            
        except Exception as e:
            self.logger.error(f"风险特征提取失败: {e}")
            features = [0] * self.risk_dim
        
        return np.array(features[:self.risk_dim])
    
    def _normalize_state(self, state_vector: np.ndarray) -> np.ndarray:
        """标准化状态向量"""
        # 使用tanh将值限制在[-1, 1]范围内
        return np.tanh(state_vector / 2)



class RewardFunction:
    """
    奖励函数设计 - 市场自适应的多目标优化
    
    核心设计原则：
    1. 不同市场状态下的动态奖励权重
    2. 与ML模型性能联动的奖励调整
    3. 风险调整的收益优化
    4. 防止过拟合特定市场环境
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("RL_REWARD_FUNCTION")
        
        # 基础奖励权重
        self.base_weights = {
            'profit': self.config.get("rl.reward.profit_weight", 1.0),
            'risk_penalty': self.config.get("rl.reward.risk_penalty", 0.5),
            'sharpe_bonus': self.config.get("rl.reward.sharpe_bonus", 0.3),
            'consistency_bonus': self.config.get("rl.reward.consistency_bonus", 0.2),
            'ml_alignment_bonus': self.config.get("rl.reward.ml_alignment_bonus", 0.15)
        }
        
        # 交易成本
        self.transaction_cost = self.config.get("rl.reward.transaction_cost", 0.001)
        self.slippage_cost = self.config.get("rl.reward.slippage_cost", 0.0005)
        
        # 市场状态特定的奖励调整
        self.market_state_modifiers = self._init_market_state_modifiers()
        
        # 性能追踪
        self.reward_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=100)
        self.ml_signal_alignment = deque(maxlen=100)
        
        # 动态奖励参数
        self.adaptive_params = {
            'risk_tolerance': 1.0,
            'profit_emphasis': 1.0,
            'exploration_bonus': 0.1
        }
    
    def _init_market_state_modifiers(self) -> Dict[str, Dict[str, float]]:
        """初始化市场状态特定的奖励修正系数"""
        return {
            MarketState.BULL_TREND.value: {
                'buy_bonus': 1.2, 'sell_penalty': 0.8, 'hold_neutral': 0.95,
                'risk_tolerance': 1.2, 'profit_emphasis': 1.1
            },
            MarketState.BEAR_TREND.value: {
                'buy_penalty': 0.8, 'sell_bonus': 1.2, 'hold_bonus': 1.05,
                'risk_tolerance': 0.8, 'profit_emphasis': 0.9
            },
            MarketState.HIGH_VOLATILITY.value: {
                'action_penalty': 1.1, 'hold_bonus': 1.15, 'risk_penalty': 1.3,
                'risk_tolerance': 0.7, 'profit_emphasis': 0.8
            },
            MarketState.LOW_VOLATILITY.value: {
                'action_bonus': 1.1, 'hold_penalty': 0.9, 'risk_penalty': 0.8,
                'risk_tolerance': 1.3, 'profit_emphasis': 1.2
            },
            MarketState.CRISIS.value: {
                'sell_bonus': 1.5, 'buy_penalty': 0.5, 'hold_bonus': 1.2,
                'risk_tolerance': 0.5, 'profit_emphasis': 0.7
            },
            MarketState.RECOVERY.value: {
                'buy_bonus': 1.3, 'sell_penalty': 0.7, 'hold_neutral': 1.0,
                'risk_tolerance': 1.1, 'profit_emphasis': 1.3
            },
            MarketState.SIDEWAYS.value: {
                'action_penalty': 1.05, 'hold_neutral': 1.0, 'scalp_bonus': 1.1,
                'risk_tolerance': 1.0, 'profit_emphasis': 1.0
            },
            MarketState.ACCUMULATION.value: {
                'buy_bonus': 1.15, 'sell_penalty': 0.85, 'patience_bonus': 1.1,
                'risk_tolerance': 1.1, 'profit_emphasis': 1.0
            },
            MarketState.DISTRIBUTION.value: {
                'sell_bonus': 1.15, 'buy_penalty': 0.85, 'caution_bonus': 1.1,
                'risk_tolerance': 0.9, 'profit_emphasis': 1.1
            }
        }
    
    def calculate_reward(self,
                        action: int,
                        previous_state: Dict[str, Any],
                        current_state: Dict[str, Any],
                        market_context: MarketContext,
                        ml_signal: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, float]]:
        """
        计算综合奖励
        
        Args:
            action: 执行的动作 (0=hold, 1=buy, 2=sell)
            previous_state: 前一状态信息
            current_state: 当前状态信息
            market_context: 市场上下文
            ml_signal: ML模型信号（用于对齐奖励）
        
        Returns:
            (总奖励, 奖励组成部分明细)
        """
        try:
            reward_components = {}
            
            # 1. 基础收益奖励
            profit_reward = self._calculate_profit_reward(
                action, previous_state, current_state, market_context
            )
            reward_components['profit'] = profit_reward
            
            # 2. 风险调整奖励
            risk_reward = self._calculate_risk_reward(
                action, current_state, market_context
            )
            reward_components['risk'] = risk_reward
            
            # 3. 夏普比率奖励
            sharpe_reward = self._calculate_sharpe_reward()
            reward_components['sharpe'] = sharpe_reward
            
            # 4. 一致性奖励
            consistency_reward = self._calculate_consistency_reward(action)
            reward_components['consistency'] = consistency_reward
            
            # 5. ML信号对齐奖励
            if ml_signal:
                ml_alignment_reward = self._calculate_ml_alignment_reward(
                    action, ml_signal, market_context
                )
                reward_components['ml_alignment'] = ml_alignment_reward
            else:
                reward_components['ml_alignment'] = 0.0
            
            # 6. 市场状态特定奖励
            market_specific_reward = self._calculate_market_specific_reward(
                action, current_state, market_context
            )
            reward_components['market_specific'] = market_specific_reward
            
            # 7. 交易成本惩罚
            if action != 0:  # 非持有动作
                cost_penalty = -(self.transaction_cost + self.slippage_cost)
                reward_components['transaction_cost'] = cost_penalty
            else:
                reward_components['transaction_cost'] = 0.0
            
            # 计算总奖励
            total_reward = self._aggregate_rewards(reward_components, market_context)
            
            # 记录奖励历史
            self.reward_history.append(total_reward)
            self.action_history.append(action)
            
            # 更新自适应参数
            self._update_adaptive_params(reward_components, market_context)
            
            return total_reward, reward_components
            
        except Exception as e:
            self.logger.error(f"奖励计算失败: {e}")
            return 0.0, {}
    
    def _calculate_profit_reward(self, action: int, previous_state: Dict[str, Any],
                               current_state: Dict[str, Any], 
                               market_context: MarketContext) -> float:
        """计算利润奖励"""
        previous_price = previous_state['price']
        current_price = current_state['price']
        position = previous_state['position']
        
        price_change = (current_price - previous_price) / previous_price
        
        # 基础收益
        if action == 0:  # Hold
            base_reward = position * price_change
        elif action == 1:  # Buy
            # 买入后立即产生的潜在收益
            base_reward = price_change - self.transaction_cost
        elif action == 2:  # Sell
            # 卖出避免的潜在损失
            base_reward = -price_change - self.transaction_cost
        else:
            base_reward = 0.0
        
        # 市场状态调整
        market_modifiers = self.market_state_modifiers.get(
            market_context.current_state.value, {}
        )
        
        # 根据动作和市场状态调整奖励
        if action == 0:
            modifier = market_modifiers.get('hold_neutral', 1.0)
        elif action == 1:
            modifier = market_modifiers.get('buy_bonus' if base_reward > 0 else 'buy_penalty', 1.0)
        else:
            modifier = market_modifiers.get('sell_bonus' if base_reward > 0 else 'sell_penalty', 1.0)
        
        # 应用市场状态修正和权重
        profit_reward = base_reward * modifier * self.base_weights['profit']
        profit_reward *= self.adaptive_params['profit_emphasis']
        
        return np.clip(profit_reward, -1.0, 1.0)
    
    def _calculate_risk_reward(self, action: int, current_state: Dict[str, Any],
                             market_context: MarketContext) -> float:
        """计算风险调整奖励"""
        position_ratio = current_state.get('position_ratio', 0)
        drawdown = current_state.get('drawdown', 0)
        volatility = current_state.get('volatility', 0)
        
        # 风险暴露
        risk_exposure = abs(position_ratio) * (1 + volatility)
        
        # 根据市场状态调整风险容忍度
        market_modifiers = self.market_state_modifiers.get(
            market_context.current_state.value, {}
        )
        risk_tolerance = market_modifiers.get('risk_tolerance', 1.0)
        
        # 风险惩罚
        if risk_exposure > risk_tolerance:
            risk_penalty = -self.base_weights['risk_penalty'] * (risk_exposure - risk_tolerance)
        else:
            # 适度风险给予小奖励
            risk_penalty = 0.05 * (risk_tolerance - risk_exposure)
        
        # 回撤惩罚
        if abs(drawdown) > 0.05:  # 5%回撤阈值
            risk_penalty -= self.base_weights['risk_penalty'] * abs(drawdown)
        
        # 市场压力调整
        risk_penalty *= (1 + market_context.stress_level)
        
        return np.clip(risk_penalty, -1.0, 0.5)
    
    def _calculate_sharpe_reward(self) -> float:
        """计算夏普比率奖励"""
        if len(self.reward_history) < 20:
            return 0.0
        
        recent_rewards = list(self.reward_history)[-20:]
        
        if np.std(recent_rewards) > 0:
            sharpe = np.mean(recent_rewards) / np.std(recent_rewards)
            sharpe_reward = self.base_weights['sharpe_bonus'] * np.tanh(sharpe / 2)
        else:
            sharpe_reward = 0.0
        
        return sharpe_reward
    
    def _calculate_consistency_reward(self, action: int) -> float:
        """计算一致性奖励（避免频繁切换）"""
        if len(self.action_history) < 5:
            return 0.0
        
        recent_actions = list(self.action_history)[-5:]
        
        # 计算动作切换频率
        switches = sum(1 for i in range(1, len(recent_actions)) 
                      if recent_actions[i] != recent_actions[i-1])
        
        # 频繁切换惩罚
        if switches > 3:
            consistency_penalty = -self.base_weights['consistency_bonus'] * (switches / 4)
        else:
            # 保持一致性奖励
            consistency_penalty = self.base_weights['consistency_bonus'] * (1 - switches / 4)
        
        return consistency_penalty
    
    def _calculate_ml_alignment_reward(self, action: int, ml_signal: Dict[str, Any],
                                     market_context: MarketContext) -> float:
        """计算与ML信号的对齐奖励"""
        ml_action = ml_signal.get('action', 0)
        ml_confidence = ml_signal.get('confidence', 0.5)
        
        # 记录对齐情况
        is_aligned = (action == ml_action)
        self.ml_signal_alignment.append(is_aligned)
        
        if is_aligned:
            # 对齐奖励，根据ML置信度加权
            alignment_reward = self.base_weights['ml_alignment_bonus'] * ml_confidence
            
            # 高置信度信号给予额外奖励
            if ml_confidence > 0.8:
                alignment_reward *= 1.5
        else:
            # 不对齐时，如果ML置信度低，惩罚较小
            if ml_confidence < 0.6:
                alignment_reward = -self.base_weights['ml_alignment_bonus'] * 0.5
            else:
                alignment_reward = -self.base_weights['ml_alignment_bonus'] * ml_confidence
        
        # 在某些市场状态下，允许RL有更多自主权
        if market_context.current_state in [MarketState.HIGH_VOLATILITY, MarketState.CRISIS]:
            alignment_reward *= 0.7  # 降低ML对齐的重要性
        
        return alignment_reward
    
    def _calculate_market_specific_reward(self, action: int, current_state: Dict[str, Any],
                                        market_context: MarketContext) -> float:
        """计算市场状态特定的奖励"""
        market_reward = 0.0
        
        # 横盘市场的区间交易奖励
        if market_context.current_state == MarketState.SIDEWAYS:
            if action != 0:  # 非持有动作
                price_position = current_state.get('price_position', 0.5)
                if (action == 1 and price_position < 0.3) or (action == 2 and price_position > 0.7):
                    market_reward = 0.1  # 在区间边缘交易
        
        # 趋势市场的顺势奖励
        elif market_context.current_state in [MarketState.BULL_TREND, MarketState.BEAR_TREND]:
            trend_strength = current_state.get('trend_strength', 0)
            if market_context.current_state == MarketState.BULL_TREND and action == 1:
                market_reward = 0.1 * trend_strength
            elif market_context.current_state == MarketState.BEAR_TREND and action == 2:
                market_reward = 0.1 * trend_strength
        
        # 高波动市场的谨慎奖励
        elif market_context.current_state == MarketState.HIGH_VOLATILITY:
            if action == 0:  # 持有
                market_reward = 0.15
        
        return market_reward
    
    def _aggregate_rewards(self, components: Dict[str, float], 
                         market_context: MarketContext) -> float:
        """聚合各项奖励组成部分"""
        # 基础聚合
        total = sum(components.values())
        
        # 市场状态的整体调整
        if market_context.stress_level > 0.7:
            # 高压力市场环境下，降低总体奖励以鼓励谨慎
            total *= 0.8
        
        # 探索奖励（早期训练阶段）
        if len(self.reward_history) < 100:
            exploration_bonus = self.adaptive_params['exploration_bonus'] * np.random.uniform(-0.1, 0.1)
            total += exploration_bonus
        
        # 限制奖励范围
        return np.clip(total, -2.0, 2.0)
    
    def _update_adaptive_params(self, reward_components: Dict[str, float],
                              market_context: MarketContext):
        """更新自适应参数"""
        # 根据近期表现调整风险容忍度
        if len(self.reward_history) >= 50:
            recent_performance = np.mean(list(self.reward_history)[-50:])
            
            if recent_performance > 0.1:
                # 表现良好，可以承担更多风险
                self.adaptive_params['risk_tolerance'] = min(1.5, 
                    self.adaptive_params['risk_tolerance'] * 1.01)
            elif recent_performance < -0.1:
                # 表现不佳，降低风险
                self.adaptive_params['risk_tolerance'] = max(0.5, 
                    self.adaptive_params['risk_tolerance'] * 0.99)
        
        # 根据市场状态调整利润重视程度
        market_modifiers = self.market_state_modifiers.get(
            market_context.current_state.value, {}
        )
        self.adaptive_params['profit_emphasis'] = market_modifiers.get('profit_emphasis', 1.0)
        
        # 动态调整探索奖励
        if len(self.action_history) >= 100:
            action_diversity = len(set(list(self.action_history)[-100:]))
            if action_diversity < 2:
                # 动作过于单一，增加探索
                self.adaptive_params['exploration_bonus'] = min(0.2, 
                    self.adaptive_params['exploration_bonus'] * 1.1)
            else:
                self.adaptive_params['exploration_bonus'] = max(0.01, 
                    self.adaptive_params['exploration_bonus'] * 0.95)
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """获取回合指标"""
        if not self.reward_history:
            return {}
        
        rewards = list(self.reward_history)
        
        metrics = {
            'total_reward': sum(rewards),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'sharpe_ratio': np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0,
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'positive_reward_ratio': sum(1 for r in rewards if r > 0) / len(rewards)
        }
        
        # ML对齐度
        if self.ml_signal_alignment:
            metrics['ml_alignment_rate'] = sum(self.ml_signal_alignment) / len(self.ml_signal_alignment)
        
        # 动作分布
        if self.action_history:
            actions = list(self.action_history)
            metrics['action_distribution'] = {
                'hold': actions.count(0) / len(actions),
                'buy': actions.count(1) / len(actions),
                'sell': actions.count(2) / len(actions)
            }
        
        return metrics




class TrajectoryBuffer:
    """
    轨迹缓冲区 - 支持增量学习的智能经验回放
    
    核心设计原则：
    1. 保留多样化的市场状态经验
    2. 优先级采样重要经验
    3. 防止灾难性遗忘
    4. 支持元学习的经验组织
    """
    
    def __init__(self, capacity: int = 100000, config_path: str = "config.yaml"):
        self.capacity = capacity
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("TRAJECTORY_BUFFER")
        
        # 主缓冲区
        self.buffer = deque(maxlen=capacity)
        
        # 按市场状态组织的子缓冲区
        self.market_state_buffers = {
            state.value: deque(maxlen=capacity // 10)  # 每种市场状态保留10%容量
            for state in MarketState
        }
        
        # 优先级相关
        self.priorities = deque(maxlen=capacity)
        self.priority_alpha = self.config.get("rl.replay.priority_alpha", 0.6)
        self.priority_beta = self.config.get("rl.replay.priority_beta", 0.4)
        self.priority_epsilon = 1e-6
        
        # 经验统计
        self.experience_stats = {
            'total_experiences': 0,
            'market_state_distribution': {state.value: 0 for state in MarketState},
            'reward_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'action_distribution': {0: 0, 1: 0, 2: 0}
        }
        
        # 关键经验保护（防止遗忘）
        self.protected_experiences = deque(maxlen=capacity // 20)  # 5%保护容量
        self.protection_criteria = {
            'high_reward_threshold': 0.5,
            'low_reward_threshold': -0.5,
            'rare_state_threshold': 0.05  # 少于5%的市场状态
        }
    
    def add(self, 
            state: np.ndarray, 
            action: int, 
            reward: float,
            next_state: np.ndarray, 
            done: bool,
            info: Dict[str, Any]):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            info: 额外信息（必须包含market_context）
        """
        # 提取市场上下文
        market_context = info.get('market_context')
        if not market_context:
            self.logger.warning("经验缺少市场上下文信息")
            return
        
        # 创建增强的经验
        experience = {
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'market_state': market_context.current_state.value,
            'market_confidence': market_context.state_confidence,
            'stress_level': market_context.stress_level,
            'info': info,
            'timestamp': TimeUtils.now_timestamp(),
            'priority': self._calculate_initial_priority(reward, market_context)
        }
        
        # 添加到主缓冲区
        self.buffer.append(experience)
        self.priorities.append(experience['priority'])
        
        # 添加到市场状态子缓冲区
        market_state = market_context.current_state.value
        self.market_state_buffers[market_state].append(experience)
        
        # 检查是否需要保护
        if self._should_protect_experience(experience):
            self.protected_experiences.append(experience)
        
        # 更新统计
        self._update_statistics(experience)
        
        # 动态调整优先级beta
        self._update_priority_beta()
    
    def sample(self, batch_size: int, use_priority: bool = True) -> List[Dict[str, Any]]:
        """
        采样批次数据
        
        Args:
            batch_size: 批次大小
            use_priority: 是否使用优先级采样
        
        Returns:
            采样的经验列表
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if use_priority:
            # 优先级采样
            experiences, indices, weights = self._priority_sample(batch_size)
            
            # 添加采样权重到经验中
            for exp, weight in zip(experiences, weights):
                exp['sampling_weight'] = weight
            
            return experiences
        else:
            # 均匀采样
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def sample_diverse_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        采样多样化批次（确保包含不同市场状态）
        
        这对于防止过拟合特定市场环境很重要
        """
        experiences = []
        
        # 计算每种市场状态应采样的数量
        states_with_data = [state for state, buffer in self.market_state_buffers.items() 
                           if len(buffer) > 0]
        
        if not states_with_data:
            return self.sample(batch_size, use_priority=False)
        
        samples_per_state = batch_size // len(states_with_data)
        remaining = batch_size % len(states_with_data)
        
        # 从每种市场状态采样
        for i, state in enumerate(states_with_data):
            state_buffer = self.market_state_buffers[state]
            n_samples = samples_per_state + (1 if i < remaining else 0)
            
            if len(state_buffer) >= n_samples:
                indices = np.random.choice(len(state_buffer), n_samples, replace=False)
                experiences.extend([state_buffer[idx] for idx in indices])
            else:
                # 如果该状态经验不足，全部使用
                experiences.extend(list(state_buffer))
        
        # 如果还不够，从主缓冲区补充
        if len(experiences) < batch_size:
            additional_needed = batch_size - len(experiences)
            additional = self.sample(additional_needed, use_priority=True)
            experiences.extend(additional)
        
        return experiences[:batch_size]
    
    def get_recent_trajectory(self, length: int = 100) -> List[Dict[str, Any]]:
        """获取最近的轨迹"""
        if len(self.buffer) < length:
            return list(self.buffer)
        
        return list(self.buffer)[-length:]
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """更新经验优先级（基于TD误差）"""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                priority = (abs(td_error) + self.priority_epsilon) ** self.priority_alpha
                self.priorities[idx] = priority
                
                # 同时更新缓冲区中的优先级
                if idx < len(self.buffer):
                    self.buffer[idx]['priority'] = priority
    
    def _calculate_initial_priority(self, reward: float, 
                                  market_context: MarketContext) -> float:
        """计算初始优先级"""
        # 基于奖励的优先级
        reward_priority = abs(reward) + self.priority_epsilon
        
        # 罕见市场状态加成
        state_rarity = 1.0 / (self.experience_stats['market_state_distribution'].get(
            market_context.current_state.value, 1) / max(self.experience_stats['total_experiences'], 1) + 0.1)
        
        # 高压力市场环境加成
        stress_bonus = 1.0 + market_context.stress_level
        
        # 综合优先级
        priority = reward_priority * state_rarity * stress_bonus
        
        return priority ** self.priority_alpha
    
    def _priority_sample(self, batch_size: int) -> Tuple[List[Dict], List[int], List[float]]:
        """优先级采样"""
        priorities = np.array(list(self.priorities))
        
        # 计算采样概率
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.priority_beta)
        weights /= weights.max()  # 标准化
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def _should_protect_experience(self, experience: Dict[str, Any]) -> bool:
        """判断是否应该保护该经验"""
        # 高奖励或低奖励经验
        if (experience['reward'] > self.protection_criteria['high_reward_threshold'] or
            experience['reward'] < self.protection_criteria['low_reward_threshold']):
            return True
        
        # 罕见市场状态
        state_ratio = (self.experience_stats['market_state_distribution'].get(
            experience['market_state'], 0) / max(self.experience_stats['total_experiences'], 1))
        
        if state_ratio < self.protection_criteria['rare_state_threshold']:
            return True
        
        # 高压力市场环境
        if experience['stress_level'] > 0.8:
            return True
        
        return False
    
    def _update_statistics(self, experience: Dict[str, Any]):
        """更新统计信息"""
        self.experience_stats['total_experiences'] += 1
        
        # 市场状态分布
        market_state = experience['market_state']
        self.experience_stats['market_state_distribution'][market_state] += 1
        
        # 奖励分布
        if experience['reward'] > 0.1:
            self.experience_stats['reward_distribution']['positive'] += 1
        elif experience['reward'] < -0.1:
            self.experience_stats['reward_distribution']['negative'] += 1
        else:
            self.experience_stats['reward_distribution']['neutral'] += 1
        
        # 动作分布
        self.experience_stats['action_distribution'][experience['action']] += 1
    
    def _update_priority_beta(self):
        """动态调整优先级beta（控制重要性采样的程度）"""
        # 随着经验增加，逐渐增加beta
        progress = min(self.experience_stats['total_experiences'] / 100000, 1.0)
        self.priority_beta = self.config.get("rl.replay.priority_beta", 0.4) + 0.6 * progress
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = {
            'total_size': len(self.buffer),
            'capacity_usage': len(self.buffer) / self.capacity,
            'protected_experiences': len(self.protected_experiences),
            **self.experience_stats
        }
        
        # 计算市场状态分布比例
        total = self.experience_stats['total_experiences']
        if total > 0:
            stats['market_state_ratios'] = {
                state: count / total 
                for state, count in self.experience_stats['market_state_distribution'].items()
            }
        
        # 奖励统计
        if self.buffer:
            rewards = [exp['reward'] for exp in list(self.buffer)[-1000:]]  # 最近1000个
            stats['recent_reward_stats'] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'max': max(rewards),
                'min': min(rewards)
            }
        
        return stats
    
    def save(self, filepath: str):
        """保存缓冲区（支持增量保存）"""
        try:
            save_data = {
                'buffer': self._serialize_buffer(list(self.buffer)),
                'protected_experiences': self._serialize_buffer(list(self.protected_experiences)),
                'statistics': self.experience_stats,
                'config': {
                    'capacity': self.capacity,
                    'priority_alpha': self.priority_alpha,
                    'priority_beta': self.priority_beta
                }
            }
            
            # 使用压缩保存大型缓冲区
            import gzip
            with gzip.open(filepath + '.gz', 'wt', encoding='utf-8') as f:
                json.dump(save_data, f)
            
            self.logger.info(f"缓冲区已保存: {filepath}.gz")
            
        except Exception as e:
            self.logger.error(f"保存缓冲区失败: {e}")
    
    def load(self, filepath: str):
        """加载缓冲区"""
        try:
            import gzip
            with gzip.open(filepath + '.gz', 'rt', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # 恢复缓冲区
            self.buffer.clear()
            for exp_data in save_data['buffer']:
                self.buffer.append(self._deserialize_experience(exp_data))
            
            # 恢复保护经验
            self.protected_experiences.clear()
            for exp_data in save_data['protected_experiences']:
                self.protected_experiences.append(self._deserialize_experience(exp_data))
            
            # 恢复统计信息
            self.experience_stats = save_data['statistics']
            
            # 恢复配置
            config = save_data['config']
            self.priority_alpha = config['priority_alpha']
            self.priority_beta = config['priority_beta']
            
            # 重建优先级队列
            self.priorities.clear()
            for exp in self.buffer:
                self.priorities.append(exp.get('priority', 1.0))
            
            self.logger.info(f"缓冲区已加载: {filepath}.gz")
            
        except Exception as e:
            self.logger.error(f"加载缓冲区失败: {e}")
    
    def _serialize_buffer(self, buffer: List[Dict]) -> List[Dict]:
        """序列化缓冲区数据"""
        serialized = []
        for exp in buffer:
            exp_copy = exp.copy()
            # 转换numpy数组
            exp_copy['state'] = exp['state'].tolist()
            exp_copy['next_state'] = exp['next_state'].tolist()
            serialized.append(exp_copy)
        return serialized
    
    def _deserialize_experience(self, exp_data: Dict) -> Dict:
        """反序列化经验"""
        exp = exp_data.copy()
        exp['state'] = np.array(exp['state'], dtype=np.float32)
        exp['next_state'] = np.array(exp['next_state'], dtype=np.float32)
        return exp