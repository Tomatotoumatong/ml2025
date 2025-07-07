# ml_strategy.py - ML策略执行器
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import joblib
from pathlib import Path
import json

from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils
from ml_pipeline import MLPipeline, FeatureEngineering
from market_environment import MarketEnvironmentClassifier


class TradingAction(Enum):
    """交易动作枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """交易信号数据类"""
    symbol: str
    action: TradingAction
    confidence: float
    size: float
    price: float
    model_type: str
    features: Dict[str, float]
    timestamp: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'confidence': self.confidence,
            'size': self.size,
            'price': self.price,
            'model_type': self.model_type,
            'features': self.features,
            'timestamp': self.timestamp
        }


class MLStrategy:
    """ML策略执行器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("ML_STRATEGY")
        
        # 初始化组件
        self.ml_pipeline = MLPipeline(config_path)
        self.feature_engineering = FeatureEngineering()
        self.market_classifier = MarketEnvironmentClassifier()
        
        # 策略配置
        self.confidence_threshold = self.config.get("ml_strategy.confidence_threshold", 0.65)
        self.position_sizing_method = self.config.get("ml_strategy.position_sizing", "kelly")
        self.max_position_size = self.config.get("trading.max_position_size", 0.1)
        self.use_ensemble = self.config.get("ml_strategy.use_ensemble", True)
        
        # 模型权重配置（可根据历史性能动态调整）
        self.model_weights = {
            'xgboost': 0.4,
            'lightgbm': 0.4,
            'catboost': 0.2
        }
        
        # 信号缓存
        self.signal_cache = {}
        self.prediction_history = []
        
        # 性能追踪
        self.performance_tracker = {
            'total_signals': 0,
            'correct_signals': 0,
            'accuracy': 0.0,
            'last_update': TimeUtils.now_timestamp()
        }
    
    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """生成交易信号"""
        try:
            # 1. 检查数据充足性
            if len(market_data) < 100:
                self.logger.warning(f"{symbol} 数据不足，无法生成信号")
                return None
            
            # 2. 特征工程
            features_df = self.feature_engineering.create_features(market_data, symbol)
            
            # 获取最新数据行
            latest_features = features_df.iloc[-1:].copy()
            
            # 移除非特征列
            feature_cols = [col for col in latest_features.columns 
                          if col not in ['label', 'future_return', 'timestamp', 'datetime']]
            X = latest_features[feature_cols]
            
            # 3. 获取预测
            if self.use_ensemble:
                predictions, probabilities = self.ml_pipeline.get_ensemble_prediction(X, symbol, self.model_weights)
            else:
                # 使用单个最佳模型
                predictions, probabilities = self._get_best_model_prediction(X, symbol)
            
            if len(predictions) == 0:
                self.logger.warning(f"{symbol} 没有可用的模型")
                return None
            
            # 4. 解析预测结果
            prediction = predictions[0]
            confidence = np.max(probabilities[0])
            
            # 5. 生成交易动作
            action = self._prediction_to_action(prediction)
            
            # 6. 检查置信度阈值
            if confidence < self.confidence_threshold:
                self.logger.info(f"{symbol} 置信度不足: {confidence:.2f} < {self.confidence_threshold}")
                action = TradingAction.HOLD
            
            # 7. 市场环境调整
            market_state, state_confidence = self.market_classifier.classify_market_environment(market_data)
            adjusted_confidence = self._adjust_confidence_by_market_state(
                confidence, market_state, action
            )
            
            # 8. 计算仓位大小
            position_size = self._calculate_position_size(
                adjusted_confidence,
                market_data.iloc[-1]['close'],
                market_data
            )
            
            # 9. 创建交易信号
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=adjusted_confidence,
                size=position_size,
                price=float(market_data.iloc[-1]['close']),
                model_type='ml_ensemble' if self.use_ensemble else 'ml_single',
                features=self._get_key_features(X.iloc[0].to_dict()),
                timestamp=TimeUtils.now_timestamp()
            )
            
            # 10. 缓存信号
            self.signal_cache[symbol] = signal
            self.performance_tracker['total_signals'] += 1
            
            # 11. 记录预测历史
            self._record_prediction(signal, probabilities[0])
            
            self.logger.info(f"生成信号: {signal.action.value} {symbol} @ {signal.price:.2f}, "
                           f"置信度: {signal.confidence:.2f}, 仓位: {signal.size:.4f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"信号生成失败 {symbol}: {e}")
            return None
    
    def _prediction_to_action(self, prediction: int) -> TradingAction:
        """将预测值转换为交易动作"""
        mapping = {
            0: TradingAction.SELL,
            1: TradingAction.HOLD,
            2: TradingAction.BUY
        }
        return mapping.get(prediction, TradingAction.HOLD)
    
    def _adjust_confidence_by_market_state(self, confidence: float, 
                                         market_state: str, 
                                         action: TradingAction) -> float:
        """根据市场状态调整置信度"""
        # 市场状态对不同动作的影响权重
        state_adjustments = {
            'bull_trend': {'buy': 1.1, 'sell': 0.9, 'hold': 1.0},
            'bear_trend': {'buy': 0.9, 'sell': 1.1, 'hold': 1.0},
            'sideways': {'buy': 0.95, 'sell': 0.95, 'hold': 1.05},
            'high_volatility': {'buy': 0.9, 'sell': 0.9, 'hold': 1.1},
            'low_volatility': {'buy': 1.05, 'sell': 1.05, 'hold': 0.95},
            'crisis': {'buy': 0.8, 'sell': 1.2, 'hold': 1.0},
            'recovery': {'buy': 1.15, 'sell': 0.85, 'hold': 1.0},
            'accumulation': {'buy': 1.1, 'sell': 0.8, 'hold': 1.0},
            'distribution': {'buy': 0.8, 'sell': 1.1, 'hold': 1.0}
        }
        
        adjustment = state_adjustments.get(market_state, {}).get(action.value, 1.0)
        adjusted_confidence = confidence * adjustment
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _calculate_position_size(self, confidence: float, current_price: float,
                               market_data: pd.DataFrame) -> float:
        """计算仓位大小"""
        if self.position_sizing_method == 'kelly':
            # Kelly准则
            win_rate = confidence
            win_loss_ratio = self._estimate_win_loss_ratio(market_data)
            
            # Kelly公式: f = (p * b - q) / b
            # p: 获胜概率, q: 失败概率, b: 赔率
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 限制最大25%
            
            position_size = kelly_fraction * self.max_position_size
            
        elif self.position_sizing_method == 'fixed':
            # 固定仓位
            position_size = self.max_position_size
            
        elif self.position_sizing_method == 'confidence_based':
            # 基于置信度的仓位
            position_size = confidence * self.max_position_size
            
        else:
            # 默认固定仓位
            position_size = self.max_position_size * 0.5
        
        return min(position_size, self.max_position_size)
    
    def _estimate_win_loss_ratio(self, market_data: pd.DataFrame) -> float:
        """估计盈亏比"""
        returns = market_data['close'].pct_change().dropna()
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            return avg_win / avg_loss
        
        return 1.5  # 默认盈亏比
    
    def _get_key_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """获取关键特征"""
        # 从特征重要性中选择最重要的特征
        if hasattr(self.feature_engineering, 'feature_importance_cache'):
            importance = self.feature_engineering.feature_importance_cache
            
            # 选择前10个最重要的特征
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            key_features = {}
            
            for feature_name, _ in top_features:
                if feature_name in features:
                    key_features[feature_name] = features[feature_name]
            
            return key_features
        
        # 如果没有特征重要性，返回部分默认特征
        default_features = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'atr']
        return {k: v for k, v in features.items() if k in default_features}
    
    def _get_best_model_prediction(self, X: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """获取最佳单模型预测"""
        # 根据历史性能选择最佳模型
        best_model = max(self.model_weights.items(), key=lambda x: x[1])[0]
        
        model_path = Path(f"models/ml/{symbol}_{best_model}_model.pkl")
        if not model_path.exists():
            return np.array([]), np.array([])
        
        # 加载模型
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # 准备数据
        X_selected = X[features]
        X_scaled = scaler.transform(X_selected)
        
        # 预测
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def _record_prediction(self, signal: TradingSignal, probabilities: np.ndarray):
        """记录预测历史"""
        prediction_record = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': signal.action.value,
            'confidence': signal.confidence,
            'probabilities': {
                'sell': float(probabilities[0]),
                'hold': float(probabilities[1]),
                'buy': float(probabilities[2])
            },
            'price': signal.price,
            'size': signal.size
        }
        
        self.prediction_history.append(prediction_record)
        
        # 限制历史记录大小
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-5000:]
    
    def update_performance(self, symbol: str, signal_timestamp: int, 
                         actual_return: float, signal_action: str):
        """更新策略性能"""
        try:
            # 判断信号是否正确
            is_correct = False
            if signal_action == 'buy' and actual_return > 0.002:
                is_correct = True
            elif signal_action == 'sell' and actual_return < -0.002:
                is_correct = True
            elif signal_action == 'hold' and abs(actual_return) < 0.002:
                is_correct = True
            
            if is_correct:
                self.performance_tracker['correct_signals'] += 1
            
            # 更新准确率
            total = self.performance_tracker['total_signals']
            correct = self.performance_tracker['correct_signals']
            self.performance_tracker['accuracy'] = correct / total if total > 0 else 0
            
            # 动态调整模型权重
            if total > 100 and total % 50 == 0:
                self._adjust_model_weights()
            
            self.logger.info(f"性能更新 - 准确率: {self.performance_tracker['accuracy']:.2%}")
            
        except Exception as e:
            self.logger.error(f"性能更新失败: {e}")
    
    def _adjust_model_weights(self):
        """动态调整模型权重"""
        # 基于最近的预测性能调整权重
        recent_predictions = self.prediction_history[-100:]
        
        if len(recent_predictions) < 50:
            return
        
        # TODO: 实现基于性能的权重调整逻辑
        # 这里简化处理，保持默认权重
        pass
    
    def should_retrain(self) -> bool:
        """判断是否需要重新训练"""
        # 准确率低于阈值
        if self.performance_tracker['accuracy'] < 0.45:
            return True
        
        # 距离上次更新时间过长
        last_update = self.performance_tracker['last_update']
        if TimeUtils.now_timestamp() - last_update > 86400000:  # 24小时
            return True
        
        return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """获取策略状态"""
        return {
            'performance': self.performance_tracker,
            'model_weights': self.model_weights,
            'active_signals': len(self.signal_cache),
            'confidence_threshold': self.confidence_threshold,
            'position_sizing_method': self.position_sizing_method,
            'recent_predictions': len(self.prediction_history)
        }