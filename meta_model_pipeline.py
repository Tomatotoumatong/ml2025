# meta_model_pipeline.py - 元模型管道
# =============================================================================
# 核心职责：
# 1. 根据市场状态和模型性能动态选择ML或RL模型
# 2. 实时监控和评估子模型性能
# 3. 实现平滑的模型切换机制
# 4. 管理模型集成和投票机制
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from market_environment import MarketEnvironmentClassifier, MarketState
from ml_strategy import MLStrategy, TradingSignal
from train_rl import PPOAgent, A2CAgent
from database_manager import DatabaseManager


class ModelType(Enum):
    """模型类型枚举"""
    ML_ENSEMBLE = "ml_ensemble"
    RL_PPO = "rl_ppo"
    RL_A2C = "rl_a2c"
    HYBRID = "hybrid"


@dataclass
class ModelPerformance:
    """模型性能跟踪"""
    model_type: ModelType
    market_state: MarketState
    
    # 性能指标
    accuracy: float = 0.5
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.5
    avg_return: float = 0.0
    
    # 统计信息
    num_trades: int = 0
    num_predictions: int = 0
    confidence_avg: float = 0.5
    
    # 时间窗口性能
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=100))
    performance_trend: float = 0.0  # 性能趋势（正=改善，负=恶化）
    
    # 更新时间
    last_update: int = 0
    
    def update(self, trade_result: Dict[str, Any]):
        """更新性能指标"""
        self.num_predictions += 1
        
        # 更新准确率
        if trade_result.get('correct', False):
            self.accuracy = (self.accuracy * (self.num_predictions - 1) + 1) / self.num_predictions
        else:
            self.accuracy = (self.accuracy * (self.num_predictions - 1)) / self.num_predictions
        
        # 更新其他指标
        if 'return' in trade_result:
            self.recent_performance.append(trade_result['return'])
            self.avg_return = np.mean(self.recent_performance)
            
            # 计算性能趋势
            if len(self.recent_performance) >= 20:
                recent = list(self.recent_performance)[-20:]
                older = list(self.recent_performance)[-40:-20] if len(self.recent_performance) >= 40 else recent
                self.performance_trend = np.mean(recent) - np.mean(older)
        
        self.last_update = TimeUtils.now_timestamp()
    
    def get_score(self) -> float:
        """获取综合性能分数"""
        # 多维度加权评分
        score = (
            self.accuracy * 0.3 +
            np.tanh(self.sharpe_ratio) * 0.25 +
            (1 + self.max_drawdown) * 0.2 +  # max_drawdown是负数
            self.win_rate * 0.15 +
            np.tanh(self.avg_return * 10) * 0.1
        )
        
        # 趋势调整
        if self.performance_trend > 0:
            score *= 1.1
        elif self.performance_trend < -0.01:
            score *= 0.9
        
        return np.clip(score, 0, 1)


class MetaModelPipeline:
    """
    元模型管道 - 系统的大脑
    
    核心功能：
    1. 智能模型选择
    2. 性能监控与评估
    3. 动态权重调整
    4. 模型切换决策
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("META_MODEL_PIPELINE")
        
        # 初始化组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.ml_strategy = MLStrategy(config_path)
        self.db_manager = DatabaseManager(config_path)
        
        # RL模型容器
        self.rl_models: Dict[str, Union[PPOAgent, A2CAgent]] = {}
        
        # 元模型配置
        self.model_selection_method = self.config.get("meta_model.selection_method", "performance_based")
        self.switch_threshold = self.config.get("meta_model.switch_threshold", 0.1)
        self.evaluation_window = self.config.get("meta_model.evaluation_window", 100)
        self.use_ensemble = self.config.get("meta_model.use_ensemble", True)
        
        # 性能跟踪
        self.model_performance: Dict[Tuple[ModelType, MarketState], ModelPerformance] = {}
        self._init_performance_tracking()
        
        # 当前活跃模型
        self.active_model: ModelType = ModelType.ML_ENSEMBLE
        self.previous_model: Optional[ModelType] = None
        self.model_switch_history = deque(maxlen=100)
        
        # 模型权重（用于集成）
        self.model_weights = {
            ModelType.ML_ENSEMBLE: 0.5,
            ModelType.RL_PPO: 0.3,
            ModelType.RL_A2C: 0.2
        }
        
        # 市场状态到模型的映射（可学习）
        self.state_model_mapping = self._init_state_model_mapping()
        
        # 切换平滑度
        self.transition_period = self.config.get("meta_model.transition_period", 10)
        self.transition_counter = 0
        
        # 模型池
        self.model_pool_size = self.config.get("meta_model.model_pool_size", 5)
        self.model_versions: Dict[ModelType, deque] = {
            model_type: deque(maxlen=self.model_pool_size)
            for model_type in ModelType
        }
    
    def _init_performance_tracking(self):
        """初始化性能跟踪"""
        for model_type in ModelType:
            for market_state in MarketState:
                key = (model_type, market_state)
                self.model_performance[key] = ModelPerformance(
                    model_type=model_type,
                    market_state=market_state
                )
    
    def _init_state_model_mapping(self) -> Dict[MarketState, Dict[ModelType, float]]:
        """初始化市场状态到模型的映射"""
        # 基于先验知识的初始映射
        return {
            MarketState.BULL_TREND: {
                ModelType.ML_ENSEMBLE: 0.4,
                ModelType.RL_PPO: 0.5,
                ModelType.RL_A2C: 0.1
            },
            MarketState.BEAR_TREND: {
                ModelType.ML_ENSEMBLE: 0.5,
                ModelType.RL_PPO: 0.3,
                ModelType.RL_A2C: 0.2
            },
            MarketState.HIGH_VOLATILITY: {
                ModelType.ML_ENSEMBLE: 0.6,
                ModelType.RL_PPO: 0.2,
                ModelType.RL_A2C: 0.2
            },
            MarketState.LOW_VOLATILITY: {
                ModelType.ML_ENSEMBLE: 0.3,
                ModelType.RL_PPO: 0.5,
                ModelType.RL_A2C: 0.2
            },
            MarketState.CRISIS: {
                ModelType.ML_ENSEMBLE: 0.7,
                ModelType.RL_PPO: 0.2,
                ModelType.RL_A2C: 0.1
            },
            MarketState.RECOVERY: {
                ModelType.ML_ENSEMBLE: 0.4,
                ModelType.RL_PPO: 0.4,
                ModelType.RL_A2C: 0.2
            },
            MarketState.SIDEWAYS: {
                ModelType.ML_ENSEMBLE: 0.5,
                ModelType.RL_PPO: 0.3,
                ModelType.RL_A2C: 0.2
            },
            MarketState.ACCUMULATION: {
                ModelType.ML_ENSEMBLE: 0.4,
                ModelType.RL_PPO: 0.4,
                ModelType.RL_A2C: 0.2
            },
            MarketState.DISTRIBUTION: {
                ModelType.ML_ENSEMBLE: 0.5,
                ModelType.RL_PPO: 0.3,
                ModelType.RL_A2C: 0.2
            }
        }
    
    def load_rl_models(self, symbol: str):
        """加载RL模型"""
        model_dir = Path("models/rl")
        
        # 加载PPO模型
        ppo_path = model_dir / f"{symbol}_ppo_best.pth"
        if ppo_path.exists():
            from prepare_rl_data import StateSpace
            state_space = StateSpace()
            ppo_agent = PPOAgent(state_space.total_dim, 3)
            ppo_agent.load(str(ppo_path))
            self.rl_models['ppo'] = ppo_agent
            self.logger.info(f"PPO模型已加载: {symbol}")
        
        # 加载A2C模型
        a2c_path = model_dir / f"{symbol}_a2c_best.pth"
        if a2c_path.exists():
            from prepare_rl_data import StateSpace
            state_space = StateSpace()
            a2c_agent = A2CAgent(state_space.total_dim, 3)
            a2c_agent.load(str(a2c_path))
            self.rl_models['a2c'] = a2c_agent
            self.logger.info(f"A2C模型已加载: {symbol}")
    
    def select_model(self, market_data: pd.DataFrame, 
                    current_state: MarketState) -> ModelType:
        """
        选择最优模型
        
        策略：
        1. 基于历史性能
        2. 考虑市场状态
        3. 评估模型置信度
        4. 平滑切换
        """
        if self.model_selection_method == "performance_based":
            selected_model = self._performance_based_selection(current_state)
        elif self.model_selection_method == "adaptive":
            selected_model = self._adaptive_selection(market_data, current_state)
        elif self.model_selection_method == "ensemble":
            selected_model = ModelType.HYBRID
        else:
            selected_model = self._rule_based_selection(current_state)
        
        # 检查是否需要切换
        if selected_model != self.active_model:
            if self._should_switch_model(selected_model, current_state):
                self._execute_model_switch(selected_model, current_state)
        
        return self.active_model
    
    def _performance_based_selection(self, current_state: MarketState) -> ModelType:
        """基于性能的模型选择"""
        best_model = ModelType.ML_ENSEMBLE
        best_score = -float('inf')
        
        # 评估每个模型在当前市场状态下的性能
        for model_type in ModelType:
            if model_type == ModelType.HYBRID:
                continue
            
            key = (model_type, current_state)
            performance = self.model_performance.get(key)
            
            if performance and performance.num_predictions > 10:
                score = performance.get_score()
                
                # 考虑映射权重
                mapping_weight = self.state_model_mapping[current_state].get(model_type, 0.5)
                adjusted_score = score * mapping_weight
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_model = model_type
        
        return best_model
    
    def _adaptive_selection(self, market_data: pd.DataFrame, 
                          current_state: MarketState) -> ModelType:
        """自适应模型选择（考虑更多因素）"""
        scores = {}
        
        # 1. 历史性能评分
        for model_type in [ModelType.ML_ENSEMBLE, ModelType.RL_PPO, ModelType.RL_A2C]:
            key = (model_type, current_state)
            performance = self.model_performance.get(key)
            
            if performance:
                perf_score = performance.get_score()
            else:
                perf_score = 0.5
            
            # 2. 市场适配度评分
            market_fit = self.state_model_mapping[current_state].get(model_type, 0.5)
            
            # 3. 近期趋势评分
            trend_score = 0.5
            if performance and performance.performance_trend != 0:
                trend_score = 0.5 + np.tanh(performance.performance_trend * 10) * 0.5
            
            # 4. 模型可用性
            availability = 1.0
            if model_type in [ModelType.RL_PPO, ModelType.RL_A2C]:
                model_name = 'ppo' if model_type == ModelType.RL_PPO else 'a2c'
                if model_name not in self.rl_models:
                    availability = 0.0
            
            # 综合评分
            scores[model_type] = (
                perf_score * 0.4 +
                market_fit * 0.3 +
                trend_score * 0.2 +
                availability * 0.1
            )
        
        # 选择最高分的模型
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _rule_based_selection(self, current_state: MarketState) -> ModelType:
        """基于规则的模型选择（后备方案）"""
        rules = {
            MarketState.HIGH_VOLATILITY: ModelType.ML_ENSEMBLE,
            MarketState.CRISIS: ModelType.ML_ENSEMBLE,
            MarketState.BULL_TREND: ModelType.RL_PPO,
            MarketState.BEAR_TREND: ModelType.ML_ENSEMBLE,
            MarketState.SIDEWAYS: ModelType.RL_PPO,
            MarketState.LOW_VOLATILITY: ModelType.RL_PPO
        }
        
        return rules.get(current_state, ModelType.ML_ENSEMBLE)
    
    def _should_switch_model(self, new_model: ModelType, 
                           current_state: MarketState) -> bool:
        """判断是否应该切换模型"""
        # 获取当前和新模型的性能
        current_key = (self.active_model, current_state)
        new_key = (new_model, current_state)
        
        current_perf = self.model_performance.get(current_key)
        new_perf = self.model_performance.get(new_key)
        
        if not current_perf or not new_perf:
            return False
        
        # 性能差异必须超过阈值
        performance_gap = new_perf.get_score() - current_perf.get_score()
        
        if performance_gap < self.switch_threshold:
            return False
        
        # 避免频繁切换
        if self.model_switch_history:
            last_switch = self.model_switch_history[-1]
            time_since_switch = TimeUtils.now_timestamp() - last_switch['timestamp']
            
            # 至少等待5分钟
            if time_since_switch < 300000:
                return False
        
        return True
    
    def _execute_model_switch(self, new_model: ModelType, 
                            current_state: MarketState):
        """执行模型切换"""
        self.logger.info(f"切换模型: {self.active_model.value} -> {new_model.value} "
                        f"(市场状态: {current_state.value})")
        
        # 记录切换
        switch_record = {
            'timestamp': TimeUtils.now_timestamp(),
            'from_model': self.active_model,
            'to_model': new_model,
            'market_state': current_state,
            'reason': 'performance_based'
        }
        self.model_switch_history.append(switch_record)
        
        # 更新模型
        self.previous_model = self.active_model
        self.active_model = new_model
        self.transition_counter = 0
    
    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        生成交易信号
        
        根据选定的模型生成信号，支持模型集成
        """
        try:
            # 获取当前市场状态
            current_state, confidence = self.market_classifier.classify_market_environment(market_data)
            
            # 选择模型
            selected_model = self.select_model(market_data, current_state)
            
            # 生成信号
            if selected_model == ModelType.ML_ENSEMBLE:
                signal = self.ml_strategy.generate_signal(market_data, symbol)
                
            elif selected_model in [ModelType.RL_PPO, ModelType.RL_A2C]:
                signal = self._generate_rl_signal(
                    market_data, symbol, selected_model, current_state
                )
                
            elif selected_model == ModelType.HYBRID:
                signal = self._generate_hybrid_signal(
                    market_data, symbol, current_state
                )
            
            else:
                signal = None
            
            # 记录信号用于性能跟踪
            if signal:
                self._record_signal(signal, current_state)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            return None
    
    def _generate_rl_signal(self, market_data: pd.DataFrame, symbol: str,
                          model_type: ModelType, 
                          current_state: MarketState) -> Optional[TradingSignal]:
        """生成RL信号"""
        model_name = 'ppo' if model_type == ModelType.RL_PPO else 'a2c'
        
        if model_name not in self.rl_models:
            self.logger.warning(f"RL模型 {model_name} 未加载")
            return None
        
        try:
            # 准备状态
            from prepare_rl_data import StateSpace
            from market_environment import MarketContext
            
            state_space = StateSpace()
            
            # 创建市场上下文
            market_context = MarketContext(
                current_state=current_state,
                state_confidence=0.8,
                regime_stability=0.7,
                transition_probability={},
                stress_level=0.2,
                liquidity_score=0.9
            )
            
            # 获取ML特征（用于RL状态构建）
            ml_features = self.ml_strategy.feature_engineering.ml_feature_cache
            feature_importance = self.ml_strategy.feature_engineering.feature_importance_cache
            
            # 构建状态向量
            state = state_space.create_state_vector(
                market_data=market_data,
                portfolio_state={
                    'position_ratio': 0.5,  # TODO: 从实际portfolio获取
                    'unrealized_pnl': 0,
                    'holding_time': 0,
                    'num_trades_today': 0,
                    'win_rate': 0.5
                },
                ml_selected_features=list(ml_features) if isinstance(ml_features, dict) else ml_features,
                feature_importance=feature_importance
            )
            
            # 获取RL动作
            agent = self.rl_models[model_name]
            
            if hasattr(agent, 'select_action'):
                action, _, _ = agent.select_action(state, deterministic=True)
            else:
                action = 0
            
            # 转换为交易信号
            from ml_strategy import TradingAction as MLTradingAction
            
            action_map = {
                0: MLTradingAction.HOLD,
                1: MLTradingAction.BUY,
                2: MLTradingAction.SELL
            }
            
            signal = TradingSignal(
                symbol=symbol,
                action=action_map[action],
                confidence=0.7,  # RL模型的置信度需要另外计算
                size=0.1,
                price=float(market_data.iloc[-1]['close']),
                model_type=model_type.value,
                features={},
                timestamp=TimeUtils.now_timestamp()
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"RL信号生成失败: {e}")
            return None
    
    def _generate_hybrid_signal(self, market_data: pd.DataFrame, symbol: str,
                              current_state: MarketState) -> Optional[TradingSignal]:
        """生成混合信号（集成多个模型）"""
        signals = []
        weights = []
        
        # 获取ML信号
        ml_signal = self.ml_strategy.generate_signal(market_data, symbol)
        if ml_signal:
            signals.append(ml_signal)
            weights.append(self.model_weights[ModelType.ML_ENSEMBLE])
        
        # 获取RL信号
        for model_type in [ModelType.RL_PPO, ModelType.RL_A2C]:
            rl_signal = self._generate_rl_signal(market_data, symbol, model_type, current_state)
            if rl_signal:
                signals.append(rl_signal)
                weights.append(self.model_weights[model_type])
        
        if not signals:
            return None
        
        # 加权投票
        from ml_strategy import TradingAction as MLTradingAction
        action_votes = {
            MLTradingAction.BUY: 0,
            MLTradingAction.SELL: 0,
            MLTradingAction.HOLD: 0
        }
        
        total_confidence = 0
        
        for signal, weight in zip(signals, weights):
            action_votes[signal.action] += weight * signal.confidence
            total_confidence += signal.confidence * weight
        
        # 选择得票最高的动作
        final_action = max(action_votes.items(), key=lambda x: x[1])[0]
        
        # 计算综合置信度
        final_confidence = total_confidence / sum(weights) if sum(weights) > 0 else 0.5
        
        # 创建混合信号
        hybrid_signal = TradingSignal(
            symbol=symbol,
            action=final_action,
            confidence=final_confidence,
            size=np.mean([s.size for s in signals]),
            price=float(market_data.iloc[-1]['close']),
            model_type=ModelType.HYBRID.value,
            features={},
            timestamp=TimeUtils.now_timestamp()
        )
        
        return hybrid_signal
    
    def update_performance(self, signal: TradingSignal, trade_result: Dict[str, Any]):
        """更新模型性能"""
        # 确定市场状态
        market_state = trade_result.get('market_state', MarketState.SIDEWAYS)
        
        # 确定模型类型
        model_type_map = {
            'ml_ensemble': ModelType.ML_ENSEMBLE,
            'rl_ppo': ModelType.RL_PPO,
            'rl_a2c': ModelType.RL_A2C,
            'hybrid': ModelType.HYBRID
        }
        
        model_type = model_type_map.get(signal.model_type, ModelType.ML_ENSEMBLE)
        
        # 更新性能
        key = (model_type, market_state)
        if key in self.model_performance:
            self.model_performance[key].update(trade_result)
        
        # 动态调整模型权重
        self._update_model_weights()
        
        # 更新状态-模型映射
        self._update_state_model_mapping(market_state, model_type, trade_result)
    
    def _update_model_weights(self):
        """动态调整模型权重"""
        # 基于最近性能调整权重
        total_score = 0
        scores = {}
        
        for model_type in [ModelType.ML_ENSEMBLE, ModelType.RL_PPO, ModelType.RL_A2C]:
            model_scores = []
            
            for market_state in MarketState:
                key = (model_type, market_state)
                if key in self.model_performance:
                    perf = self.model_performance[key]
                    if perf.num_predictions > 10:
                        model_scores.append(perf.get_score())
            
            if model_scores:
                avg_score = np.mean(model_scores)
                scores[model_type] = avg_score
                total_score += avg_score
        
        # 更新权重
        if total_score > 0:
            for model_type, score in scores.items():
                self.model_weights[model_type] = score / total_score
    
    def _update_state_model_mapping(self, market_state: MarketState,
                                  model_type: ModelType,
                                  trade_result: Dict[str, Any]):
        """更新状态-模型映射"""
        # 使用简单的在线学习更新映射权重
        learning_rate = 0.01
        
        if trade_result.get('correct', False):
            # 增强当前映射
            current_weight = self.state_model_mapping[market_state].get(model_type, 0.5)
            new_weight = current_weight + learning_rate * (1 - current_weight)
            self.state_model_mapping[market_state][model_type] = new_weight
            
            # 归一化
            total = sum(self.state_model_mapping[market_state].values())
            for mt in self.state_model_mapping[market_state]:
                self.state_model_mapping[market_state][mt] /= total
    
    def _record_signal(self, signal: TradingSignal, market_state: MarketState):
        """记录信号用于后续性能评估"""
        # TODO: 实现信号记录逻辑
        pass
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        status = {
            'active_model': self.active_model.value,
            'model_weights': {k.value: v for k, v in self.model_weights.items()},
            'recent_switches': len(self.model_switch_history),
            'performance_summary': {}
        }
        
        # 添加性能摘要
        for (model_type, market_state), perf in self.model_performance.items():
            if perf.num_predictions > 0:
                key = f"{model_type.value}_{market_state.value}"
                status['performance_summary'][key] = {
                    'score': perf.get_score(),
                    'accuracy': perf.accuracy,
                    'num_predictions': perf.num_predictions
                }
        
        return status
    
    def save_state(self, filepath: str):
        """保存元模型状态"""
        state = {
            'model_performance': {
                f"{k[0].value}_{k[1].value}": {
                    'accuracy': v.accuracy,
                    'sharpe_ratio': v.sharpe_ratio,
                    'num_predictions': v.num_predictions,
                    'recent_performance': list(v.recent_performance)
                }
                for k, v in self.model_performance.items()
            },
            'model_weights': {k.value: v for k, v in self.model_weights.items()},
            'state_model_mapping': {
                state.value: {mt.value: w for mt, w in mapping.items()}
                for state, mapping in self.state_model_mapping.items()
            },
            'active_model': self.active_model.value,
            'switch_history': list(self.model_switch_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"元模型状态已保存: {filepath}")
    
    def load_state(self, filepath: str):
        """加载元模型状态"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # 恢复性能数据
            for key_str, perf_data in state['model_performance'].items():
                model_type_str, market_state_str = key_str.split('_')
                model_type = ModelType(model_type_str)
                market_state = MarketState(market_state_str)
                
                key = (model_type, market_state)
                self.model_performance[key].accuracy = perf_data['accuracy']
                self.model_performance[key].sharpe_ratio = perf_data['sharpe_ratio']
                self.model_performance[key].num_predictions = perf_data['num_predictions']
                
                if 'recent_performance' in perf_data:
                    self.model_performance[key].recent_performance = deque(
                        perf_data['recent_performance'],
                        maxlen=100
                    )
            
            # 恢复模型权重
            self.model_weights = {
                ModelType(k): v for k, v in state['model_weights'].items()
            }
            
            # 恢复映射
            self.state_model_mapping = {
                MarketState(state_str): {
                    ModelType(mt_str): w for mt_str, w in mapping.items()
                }
                for state_str, mapping in state['state_model_mapping'].items()
            }
            
            self.logger.info(f"元模型状态已加载: {filepath}")
            
        except Exception as e:
            self.logger.error(f"加载元模型状态失败: {e}")