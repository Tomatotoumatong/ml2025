# train_meta_model.py - 元模型训练
# =============================================================================
# 核心职责：
# 1. 学习市场状态到最优模型的映射
# 2. 训练模型性能预测器
# 3. 优化模型切换策略
# 4. 实现元学习算法
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json
from scipy import stats

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from market_environment import MarketEnvironmentClassifier, MarketState
from meta_model_pipeline import ModelType, ModelPerformance
from database_manager import DatabaseManager
from technical_indicators import TechnicalIndicators


class MetaModel(nn.Module):
    """
    元模型神经网络
    
    输入：市场特征 + 历史性能
    输出：每个子模型的预期性能
    """
    
    def __init__(self, input_dim: int, num_models: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        # 市场状态编码器
        self.market_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 性能预测器（每个模型一个头）
        self.performance_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 输出0-1之间的性能分数
            ) for _ in range(num_models)
        ])
        
        # 置信度预测器
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, market_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            (模型性能预测, 预测置信度)
        """
        # 编码市场特征
        encoded = self.market_encoder(market_features)
        
        # 预测每个模型的性能
        performances = []
        for head in self.performance_heads:
            perf = head(encoded)
            performances.append(perf)
        
        performances = torch.cat(performances, dim=-1)
        
        # 预测置信度
        confidence = self.confidence_head(encoded)
        
        return performances, confidence


class MetaModelTrainer:
    """
    元模型训练器
    
    核心功能：
    1. 收集训练数据
    2. 训练性能预测模型
    3. 优化切换策略
    4. 持续学习和适应
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("META_MODEL_TRAINER")
        
        # 初始化组件
        self.market_classifier = MarketEnvironmentClassifier()
        self.technical_indicators = TechnicalIndicators(config_path)
        self.db_manager = DatabaseManager(config_path)
        
        # 训练配置
        self.learning_rate = self.config.get("meta_model.learning_rate", 0.001)
        self.batch_size = self.config.get("meta_model.batch_size", 32)
        self.num_epochs = self.config.get("meta_model.num_epochs", 100)
        self.validation_split = self.config.get("meta_model.validation_split", 0.2)
        
        # 特征维度
        self.market_feature_dim = 50  # 市场特征维度
        self.performance_history_dim = 20  # 历史性能特征维度
        self.input_dim = self.market_feature_dim + self.performance_history_dim
        
        # 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_model = MetaModel(self.input_dim, num_models=3).to(self.device)
        self.optimizer = optim.Adam(self.meta_model.parameters(), lr=self.learning_rate)
        self.scaler = StandardScaler()
        
        # 训练数据缓存
        self.training_buffer = deque(maxlen=10000)
        self.experience_pool = defaultdict(list)  # 按市场状态组织
        
        # 模型保存路径
        self.model_dir = Path("models/meta")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
    
    def collect_training_data(self, 
                            market_data: pd.DataFrame,
                            model_performances: Dict[Tuple[ModelType, MarketState], ModelPerformance],
                            actual_results: Dict[str, Any]):
        """
        收集训练数据
        
        Args:
            market_data: 市场数据
            model_performances: 模型性能记录
            actual_results: 实际交易结果
        """
        try:
            # 提取市场特征
            market_features = self._extract_market_features(market_data)
            
            # 获取当前市场状态
            current_state, _ = self.market_classifier.classify_market_environment(market_data)
            
            # 提取性能特征
            performance_features = self._extract_performance_features(
                model_performances, current_state
            )
            
            # 组合特征
            features = np.concatenate([market_features, performance_features])
            
            # 获取实际性能标签
            labels = self._extract_performance_labels(actual_results)
            
            # 创建训练样本
            sample = {
                'features': features,
                'labels': labels,
                'market_state': current_state,
                'timestamp': TimeUtils.now_timestamp()
            }
            
            # 添加到缓冲区
            self.training_buffer.append(sample)
            self.experience_pool[current_state].append(sample)
            
            self.logger.debug(f"收集训练样本 - 市场状态: {current_state.value}")
            
        except Exception as e:
            self.logger.error(f"收集训练数据失败: {e}")
    
    def _extract_market_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """提取市场特征"""
        features = []
        
        try:
            # 价格特征
            close_prices = market_data['close'].values[-50:]
            returns = np.diff(close_prices) / close_prices[:-1]
            
            features.extend([
                np.mean(returns),
                np.std(returns),
                np.max(returns),
                np.min(returns),
                returns[-1] if len(returns) > 0 else 0
            ])
            
            # 技术指标特征
            if 'rsi' in market_data.columns:
                features.append(market_data['rsi'].iloc[-1] / 100)
            else:
                features.append(0.5)
            
            if 'macd' in market_data.columns:
                features.append(np.tanh(market_data['macd'].iloc[-1]))
            else:
                features.append(0.0)
            
            if 'bb_position' in market_data.columns:
                features.append(market_data['bb_position'].iloc[-1])
            else:
                features.append(0.5)
            
            # 波动率特征
            if len(returns) >= 20:
                vol_5 = np.std(returns[-5:]) * np.sqrt(252)
                vol_20 = np.std(returns[-20:]) * np.sqrt(252)
                features.extend([vol_5, vol_20, vol_5 / vol_20 if vol_20 > 0 else 1])
            else:
                features.extend([0.02, 0.02, 1.0])
            
            # 市场微结构特征
            if 'volume' in market_data.columns:
                volumes = market_data['volume'].values[-20:]
                vol_ratio = volumes[-1] / np.mean(volumes) if len(volumes) > 0 else 1
                features.append(vol_ratio)
            else:
                features.append(1.0)
            
            # 趋势特征
            if len(close_prices) >= 20:
                # 线性回归斜率
                x = np.arange(20)
                slope = np.polyfit(x, close_prices[-20:], 1)[0]
                normalized_slope = np.tanh(slope / close_prices[-20])
                features.append(normalized_slope)
                
                # 价格相对位置
                price_position = (close_prices[-1] - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices))
                features.append(price_position)
            else:
                features.extend([0.0, 0.5])
            
            # 动量特征
            if len(close_prices) >= 10:
                momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
                features.append(momentum)
            else:
                features.append(0.0)
            
            # 市场状态特征（one-hot编码）
            current_state, confidence = self.market_classifier.classify_market_environment(market_data)
            state_encoding = [0.0] * 9
            state_index = list(MarketState).index(current_state)
            state_encoding[state_index] = confidence
            features.extend(state_encoding)
            
            # 填充到固定维度
            while len(features) < self.market_feature_dim:
                features.append(0.0)
            
            return np.array(features[:self.market_feature_dim], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"市场特征提取失败: {e}")
            return np.zeros(self.market_feature_dim, dtype=np.float32)
    
    def _extract_performance_features(self, 
                                    model_performances: Dict[Tuple[ModelType, MarketState], ModelPerformance],
                                    current_state: MarketState) -> np.ndarray:
        """提取性能特征"""
        features = []
        
        # 对每个模型提取特征
        for model_type in [ModelType.ML_ENSEMBLE, ModelType.RL_PPO, ModelType.RL_A2C]:
            key = (model_type, current_state)
            
            if key in model_performances:
                perf = model_performances[key]
                
                features.extend([
                    perf.accuracy,
                    perf.sharpe_ratio / 2,  # 标准化
                    perf.win_rate,
                    perf.avg_return * 10,  # 放大
                    perf.performance_trend * 10,
                    min(perf.num_predictions / 100, 1.0)  # 经验丰富度
                ])
            else:
                features.extend([0.5, 0, 0.5, 0, 0, 0])
        
        # 添加跨市场状态的性能统计
        for model_type in [ModelType.ML_ENSEMBLE, ModelType.RL_PPO, ModelType.RL_A2C]:
            cross_state_perfs = []
            for state in MarketState:
                key = (model_type, state)
                if key in model_performances and model_performances[key].num_predictions > 10:
                    cross_state_perfs.append(model_performances[key].get_score())
            
            if cross_state_perfs:
                features.append(np.mean(cross_state_perfs))
            else:
                features.append(0.5)
        
        # 确保长度正确
        features = features[:self.performance_history_dim]
        while len(features) < self.performance_history_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_performance_labels(self, actual_results: Dict[str, Any]) -> np.ndarray:
        """提取性能标签"""
        # 标签格式：[ML性能, PPO性能, A2C性能]
        labels = []
        
        for model_type in ['ml_ensemble', 'rl_ppo', 'rl_a2c']:
            if model_type in actual_results:
                # 综合性能评分
                result = actual_results[model_type]
                
                # 多维度性能评分
                accuracy_score = result.get('accuracy', 0.5)
                return_score = np.tanh(result.get('return', 0) * 10)
                sharpe_score = np.tanh(result.get('sharpe_ratio', 0))
                drawdown_score = 1 + result.get('max_drawdown', 0)  # drawdown是负数
                
                # 加权综合
                score = (
                    accuracy_score * 0.3 +
                    return_score * 0.3 +
                    sharpe_score * 0.2 +
                    drawdown_score * 0.2
                )
                
                labels.append(np.clip(score, 0, 1))
            else:
                labels.append(0.5)
        
        return np.array(labels, dtype=np.float32)
    
    def train(self, num_epochs: Optional[int] = None):
        """训练元模型"""
        if len(self.training_buffer) < 100:
            self.logger.warning("训练数据不足")
            return
        
        num_epochs = num_epochs or self.num_epochs
        
        self.logger.info(f"开始训练元模型 - 样本数: {len(self.training_buffer)}")
        
        # 准备数据
        X, y = self._prepare_training_data()
        
        # 分割训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 训练循环
        self.meta_model.train()
        
        for epoch in range(num_epochs):
            # 打乱数据
            perm = torch.randperm(X_train_tensor.size(0))
            X_train_tensor = X_train_tensor[perm]
            y_train_tensor = y_train_tensor[perm]
            
            # 批次训练
            total_loss = 0
            num_batches = 0
            
            for i in range(0, X_train_tensor.size(0), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                # 前向传播
                predictions, confidence = self.meta_model(batch_X)
                
                # 计算损失
                prediction_loss = nn.MSELoss()(predictions, batch_y)
                
                # 置信度损失（预测误差越大，置信度应该越低）
                errors = torch.abs(predictions - batch_y).mean(dim=1, keepdim=True)
                confidence_target = 1 - errors
                confidence_loss = nn.MSELoss()(confidence, confidence_target)
                
                # 总损失
                loss = prediction_loss + 0.1 * confidence_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # 验证
            if epoch % 10 == 0:
                val_loss = self._validate(X_val_tensor, y_val_tensor)
                
                self.training_history['loss'].append(total_loss / num_batches)
                self.training_history['val_loss'].append(val_loss)
                
                self.logger.info(f"Epoch {epoch}/{num_epochs} - "
                               f"Loss: {total_loss/num_batches:.4f}, "
                               f"Val Loss: {val_loss:.4f}")
        
        # 保存模型
        self._save_model()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        X = []
        y = []
        
        for sample in self.training_buffer:
            X.append(sample['features'])
            y.append(sample['labels'])
        
        return np.array(X), np.array(y)
    
    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """验证模型"""
        self.meta_model.eval()
        
        with torch.no_grad():
            predictions, _ = self.meta_model(X_val)
            val_loss = nn.MSELoss()(predictions, y_val)
        
        self.meta_model.train()
        return val_loss.item()
    
    def predict_performance(self, market_data: pd.DataFrame,
                          model_performances: Dict[Tuple[ModelType, MarketState], ModelPerformance]
                          ) -> Dict[ModelType, float]:
        """预测各模型性能"""
        try:
            # 提取特征
            current_state, _ = self.market_classifier.classify_market_environment(market_data)
            market_features = self._extract_market_features(market_data)
            perf_features = self._extract_performance_features(model_performances, current_state)
            
            features = np.concatenate([market_features, perf_features])
            features_scaled = self.scaler.transform([features])
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # 预测
            self.meta_model.eval()
            with torch.no_grad():
                predictions, confidence = self.meta_model(features_tensor)
            
            predictions = predictions.cpu().numpy()[0]
            confidence = confidence.cpu().item()
            
            # 返回预测结果
            return {
                ModelType.ML_ENSEMBLE: predictions[0] * confidence,
                ModelType.RL_PPO: predictions[1] * confidence,
                ModelType.RL_A2C: predictions[2] * confidence
            }
            
        except Exception as e:
            self.logger.error(f"性能预测失败: {e}")
            return {
                ModelType.ML_ENSEMBLE: 0.5,
                ModelType.RL_PPO: 0.5,
                ModelType.RL_A2C: 0.5
            }
    
    def optimize_switching_strategy(self):
        """优化切换策略"""
        # 分析切换历史，找出最优切换阈值
        switching_data = []
        
        # 从经验池中提取切换相关数据
        for state, experiences in self.experience_pool.items():
            for i in range(1, len(experiences)):
                prev_exp = experiences[i-1]
                curr_exp = experiences[i]
                
                # 计算性能差异
                perf_diff = curr_exp['labels'] - prev_exp['labels']
                
                switching_data.append({
                    'state': state,
                    'perf_diff': perf_diff,
                    'timestamp': curr_exp['timestamp']
                })
        
        if not switching_data:
            return
        
        # 分析最优切换阈值
        # 计算不同阈值下的收益
        thresholds = np.linspace(0.05, 0.3, 20)
        threshold_scores = []
        
        for threshold in thresholds:
            score = self._evaluate_threshold(switching_data, threshold)
            threshold_scores.append(score)
        
        # 选择最优阈值
        optimal_idx = np.argmax(threshold_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        self.logger.info(f"优化切换阈值: {optimal_threshold:.3f}")
        
        # 更新配置
        self.config.set("meta_model.switch_threshold", float(optimal_threshold))
    
    def _evaluate_threshold(self, switching_data: List[Dict], threshold: float) -> float:
        """评估切换阈值"""
        total_benefit = 0
        switch_count = 0
        
        for data in switching_data:
            max_diff = np.max(np.abs(data['perf_diff']))
            
            if max_diff > threshold:
                # 模拟切换
                switch_count += 1
                # 收益 = 性能提升 - 切换成本
                benefit = max_diff - 0.01  # 假设切换成本为0.01
                total_benefit += benefit
        
        # 考虑切换频率的惩罚
        if switch_count > len(switching_data) * 0.3:  # 切换过于频繁
            total_benefit *= 0.8
        
        return total_benefit
    
    def _save_model(self):
        """保存模型"""
        try:
            model_path = self.model_dir / "meta_model.pth"
            scaler_path = self.model_dir / "meta_scaler.pkl"
            
            # 保存模型
            torch.save({
                'model_state_dict': self.meta_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'config': {
                    'input_dim': self.input_dim,
                    'num_models': 3
                }
            }, model_path)
            
            # 保存标准化器
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info(f"元模型已保存: {model_path}")
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
    
    def load_model(self):
        """加载模型"""
        try:
            model_path = self.model_dir / "meta_model.pth"
            scaler_path = self.model_dir / "meta_scaler.pkl"
            
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.meta_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_history = checkpoint.get('training_history', {})
                
                self.logger.info("元模型已加载")
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计"""
        stats = {
            'total_samples': len(self.training_buffer),
            'samples_by_state': {},
            'training_history': self.training_history,
            'model_parameters': sum(p.numel() for p in self.meta_model.parameters())
        }
        
        # 统计每个市场状态的样本数
        for state in MarketState:
            stats['samples_by_state'][state.value] = len(self.experience_pool[state])
        
        # 计算样本分布均衡度
        if stats['samples_by_state']:
            sample_counts = list(stats['samples_by_state'].values())
            stats['sample_balance'] = 1 - (np.std(sample_counts) / np.mean(sample_counts))
        
        return stats