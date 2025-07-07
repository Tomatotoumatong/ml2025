# ml_pipeline.py - ML流水线控制器
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from logger import TradingLogger
from utils import ConfigManager, TimeUtils, DataUtils
from technical_indicators import TechnicalIndicators
from market_environment import MarketEnvironmentClassifier


class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self):
        self.logger = TradingLogger().get_logger("FEATURE_ENGINEERING")
        self.technical_indicators = TechnicalIndicators()
        self.market_classifier = MarketEnvironmentClassifier()
        
        # 特征配置
        self.feature_config = {
            'price_features': True,
            'technical_features': True,
            'market_regime_features': True,
            'microstructure_features': True,
            'time_features': True
        }
        
        # 特征重要性缓存
        self.feature_importance_cache = {}
    
    def create_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """创建所有特征"""
        features_df = df.copy()
        
        try:
            # 1. 价格特征
            if self.feature_config['price_features']:
                features_df = self._create_price_features(features_df)
            
            # 2. 技术指标特征
            if self.feature_config['technical_features']:
                features_df = self.technical_indicators.get_all_indicators(features_df)
            
            # 3. 市场制度特征
            if self.feature_config['market_regime_features']:
                features_df = self._create_market_regime_features(features_df)
            
            # 4. 微观结构特征
            if self.feature_config['microstructure_features']:
                features_df = self._create_microstructure_features(features_df)
            
            # 5. 时间特征
            if self.feature_config['time_features']:
                features_df = self._create_time_features(features_df)
            
            # 6. 交互特征
            features_df = self._create_interaction_features(features_df)
            
            # 记录特征数量
            self.logger.info(f"{symbol} 特征工程完成 - 特征数: {len(features_df.columns)}")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"特征创建失败: {e}")
            return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格相关特征"""
        # 收益率特征
        for lag in [1, 5, 10, 20]:
            df[f'return_{lag}'] = df['close'].pct_change(lag)
            df[f'log_return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        
        # 价格位置特征
        df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min())
        
        # 价格动量特征
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # 高低价特征
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']
        
        # 跳空缺口
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        return df
    
    def _create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建市场制度特征"""
        # 提取市场环境特征向量
        market_features = self.market_classifier.extract_market_features(df)
        
        # 将特征向量展开为列
        feature_names = [
            'trend_strength', 'trend_consistency', 'ma_alignment', 'breakout_strength',
            'momentum_strength', 'momentum_consistency', 'overbought_oversold',
            'volatility_regime', 'volatility_trend', 'volume_trend', 'volume_consistency',
            'market_efficiency', 'mean_reversion', 'market_stress'
        ]
        
        # 确保特征数量匹配
        num_features = min(len(market_features), len(feature_names))
        for i in range(num_features):
            df[feature_names[i]] = market_features[i]
        
        # 市场状态分类
        state, confidence = self.market_classifier.classify_market_environment(df)
        df['market_state'] = state.value
        df['market_state_confidence'] = confidence
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建微观结构特征"""
        if 'volume' not in df.columns:
            return df
        
        # 成交量特征
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()
        
        # 价量关系
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # Amihud非流动性指标
        df['illiquidity'] = abs(df['return_1']) / df['volume']
        df['illiquidity_ma'] = df['illiquidity'].rolling(20).mean()
        
        # Kyle's Lambda (简化版)
        df['kyle_lambda'] = abs(df['return_1']) / np.sqrt(df['volume'])
        
        # 买卖压力（如果有bid/ask数据）
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
            df['spread'] = df['ask_price'] - df['bid_price'] if 'ask_price' in df.columns else 0
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征"""
        if 'timestamp' not in df.columns:
            return df
        
        # 转换时间戳
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 时间周期特征
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 交易时段特征
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        # RSI与价格位置交互
        if 'rsi' in df.columns and 'price_position_20' in df.columns:
            df['rsi_price_interaction'] = df['rsi'] * df['price_position_20']
        
        # 成交量与波动率交互
        if 'volume_ratio' in df.columns and 'atr' in df.columns:
            df['volume_volatility_interaction'] = df['volume_ratio'] * df['atr']
        
        # MACD与ADX交互（趋势强度）
        if 'macd' in df.columns and 'adx' in df.columns:
            df['trend_strength_interaction'] = df['macd'] * df['adx']
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'importance', top_k: int = 50) -> List[str]:
        """特征选择"""
        try:
            if method == 'importance':
                # 使用LightGBM计算特征重要性
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1
                )
                lgb_model.fit(X, y)
                
                # 获取特征重要性
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': lgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                selected_features = importance.head(top_k)['feature'].tolist()
                
                # 缓存特征重要性
                self.feature_importance_cache = importance.set_index('feature')['importance'].to_dict()
                
            elif method == 'mutual_info':
                # 互信息特征选择
                selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
                selector.fit(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                
            elif method == 'statistical':
                # 统计特征选择
                selector = SelectKBest(score_func=f_classif, k=top_k)
                selector.fit(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
            
            else:
                # 默认使用方差阈值
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(X)
                selected_features = X.columns[selector.get_support()].tolist()[:top_k]
            
            self.logger.info(f"特征选择完成 - 方法: {method}, 选择特征数: {len(selected_features)}")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"特征选择失败: {e}")
            return X.columns.tolist()[:top_k]


class MLPipeline:
    """机器学习流水线"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("ML_PIPELINE")
        self.feature_engineering = FeatureEngineering()
        
        # 模型存储路径
        self.model_dir = Path("models/ml")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置
        self.models = {
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'multi:softprob',
                    'num_class': 3,  # buy, sell, hold
                    'eval_metric': 'mlogloss',
                    'use_label_encoder': False,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 300,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'catboost': {
                'model': CatBoostClassifier,
                'params': {
                    'iterations': 300,
                    'depth': 8,
                    'learning_rate': 0.1,
                    'loss_function': 'MultiClass',
                    'classes_count': 3,
                    'random_state': 42,
                    'verbose': False
                }
            }
        }
        
        # 训练配置
        self.cv_splits = self.config.get("ml.cv_splits", 5)
        self.validation_size = self.config.get("ml.validation_size", 0.2)
        self.feature_selection_method = self.config.get("ml.feature_selection_method", "importance")
        self.top_features = self.config.get("ml.top_features", 50)
        
        # 缩放器
        self.scaler = RobustScaler()
        
        # 训练历史
        self.training_history = []
    
    def prepare_training_data(self, df: pd.DataFrame, symbol: str, 
                            lookback: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        try:
            # 特征工程
            features_df = self.feature_engineering.create_features(df, symbol)
            
            # 创建标签
            features_df['future_return'] = features_df['close'].shift(-1) / features_df['close'] - 1
            
            # 标签分类：上涨(2)、下跌(0)、横盘(1)
            threshold = 0.002  # 0.2%阈值
            features_df['label'] = 1  # 默认横盘
            features_df.loc[features_df['future_return'] > threshold, 'label'] = 2  # 上涨
            features_df.loc[features_df['future_return'] < -threshold, 'label'] = 0  # 下跌
            
            # 移除无效行
            features_df = features_df.dropna()
            
            # 分离特征和标签
            feature_cols = [col for col in features_df.columns 
                          if col not in ['label', 'future_return', 'timestamp', 'datetime']]
            
            X = features_df[feature_cols]
            y = features_df['label']
            
            self.logger.info(f"训练数据准备完成 - 样本数: {len(X)}, 特征数: {len(feature_cols)}")
            self.logger.info(f"标签分布 - 下跌: {(y==0).sum()}, 横盘: {(y==1).sum()}, 上涨: {(y==2).sum()}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"训练数据准备失败: {e}")
            raise
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    symbol: str) -> Dict[str, Any]:
        """训练多个模型"""
        try:
            results = {}
            
            # 特征选择
            selected_features = self.feature_engineering.select_features(
                X, y, 
                method=self.feature_selection_method,
                top_k=self.top_features
            )
            X_selected = X[selected_features]
            
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X_selected)
            X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            
            # 训练每个模型
            for model_name, model_config in self.models.items():
                self.logger.info(f"训练 {model_name} 模型...")
                
                # 创建模型实例
                model = model_config['model'](**model_config['params'])
                
                # 交叉验证
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # 训练
                    model.fit(X_train, y_train)
                    
                    # 验证
                    y_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    cv_scores.append(accuracy)
                
                # 在全量数据上训练最终模型
                model.fit(X_scaled, y)
                
                # 保存模型
                model_path = self.model_dir / f"{symbol}_{model_name}_model.pkl"
                joblib.dump({
                    'model': model,
                    'scaler': self.scaler,
                    'features': selected_features,
                    'feature_importance': self.feature_engineering.feature_importance_cache
                }, model_path)
                
                # 记录结果
                results[model_name] = {
                    'cv_scores': cv_scores,
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores),
                    'model_path': str(model_path)
                }
                
                self.logger.info(f"{model_name} CV准确率: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            
            # 保存训练历史
            self.training_history.append({
                'timestamp': TimeUtils.now_timestamp(),
                'symbol': symbol,
                'results': results,
                'selected_features': selected_features
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series, 
                       symbol: str) -> Dict[str, Dict[str, float]]:
        """评估模型性能"""
        evaluation_results = {}
        
        for model_name in self.models.keys():
            model_path = self.model_dir / f"{symbol}_{model_name}_model.pkl"
            
            if not model_path.exists():
                continue
            
            # 加载模型
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            
            # 准备测试数据
            X_test_selected = X_test[features]
            X_test_scaled = scaler.transform(X_test_selected)
            
            # 预测
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # 计算指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # 计算每类的准确率
            for i in range(3):
                mask = y_test == i
                if mask.sum() > 0:
                    class_acc = accuracy_score(y_test[mask], y_pred[mask])
                    metrics[f'class_{i}_accuracy'] = class_acc
            
            evaluation_results[model_name] = metrics
            
            self.logger.info(f"{model_name} 评估结果: {metrics}")
        
        return evaluation_results
    
    def get_ensemble_prediction(self, X: pd.DataFrame, symbol: str, 
                              weights: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """集成预测"""
        if weights is None:
            weights = {'xgboost': 0.4, 'lightgbm': 0.4, 'catboost': 0.2}
        
        predictions = []
        probabilities = []
        
        for model_name, weight in weights.items():
            model_path = self.model_dir / f"{symbol}_{model_name}_model.pkl"
            
            if not model_path.exists():
                continue
            
            # 加载模型
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            
            # 准备数据
            X_selected = X[features]
            X_scaled = scaler.transform(X_selected)
            
            # 预测
            pred_proba = model.predict_proba(X_scaled)
            probabilities.append(pred_proba * weight)
        
        if not probabilities:
            return np.array([]), np.array([])
        
        # 加权平均
        ensemble_proba = np.sum(probabilities, axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba
    
    def update_feature_importance(self, symbol: str, performance_metrics: Dict[str, float]):
        """根据性能更新特征重要性"""
        try:
            # 根据模型性能调整特征权重
            if performance_metrics.get('accuracy', 0) < 0.45:
                self.logger.warning(f"{symbol} 模型性能低于阈值，触发特征重新评估")
                
                # 降低当前特征的权重
                for feature, importance in self.feature_engineering.feature_importance_cache.items():
                    self.feature_engineering.feature_importance_cache[feature] *= 0.9
                
                # 标记需要重新训练
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"特征重要性更新失败: {e}")
            return False