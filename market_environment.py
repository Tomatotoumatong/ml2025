
# market_environment.py - 市场环境识别
# =============================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from enum import Enum
from logger import TradingLogger
from utils import ConfigManager
from technical_indicators import TechnicalIndicators
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple, Any
from scipy.signal import argrelextrema
from utils import ConfigManager, TimeUtils

class MarketState(Enum):
    """市场状态枚举"""
    BULL_TREND = "bull_trend"           # 牛市趋势
    BEAR_TREND = "bear_trend"           # 熊市趋势
    SIDEWAYS = "sideways"               # 横盘震荡
    HIGH_VOLATILITY = "high_volatility" # 高波动
    LOW_VOLATILITY = "low_volatility"   # 低波动
    CRISIS = "crisis"                   # 危机模式
    RECOVERY = "recovery"               # 恢复模式
    ACCUMULATION = "accumulation"       # 积累模式
    DISTRIBUTION = "distribution"       # 分发模式


class MarketEnvironmentClassifier:
    """市场环境分类器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("MARKET_ENVIRONMENT")
        self.technical_indicators = TechnicalIndicators(config_path)
        
        # 模型存储路径
        self.model_path = Path("models/market_environment_classifier.pkl")
        self.model_path.parent.mkdir(exist_ok=True)
        
        # 分类器组件
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.classifier = KMeans(n_clusters=9, random_state=42)
        
        # 市场状态映射
        self.state_mapping = {
            0: MarketState.BULL_TREND,
            1: MarketState.BEAR_TREND,
            2: MarketState.SIDEWAYS,
            3: MarketState.HIGH_VOLATILITY,
            4: MarketState.LOW_VOLATILITY,
            5: MarketState.CRISIS,
            6: MarketState.RECOVERY,
            7: MarketState.ACCUMULATION,
            8: MarketState.DISTRIBUTION
        }
        
        # 特征权重
        self.feature_weights = {
            'trend_features': 0.3,      # 趋势特征
            'momentum_features': 0.25,   # 动量特征
            'volatility_features': 0.2,  # 波动率特征
            'volume_features': 0.15,     # 成交量特征
            'structure_features': 0.1    # 市场结构特征
        }
        
        # 加载预训练模型
        self.load_model()
    
    def extract_market_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取市场特征"""
        try:
            # 计算技术指标
            df_with_indicators = self.technical_indicators.get_all_indicators(df)
            
            features = []
            
            # 1. 趋势特征
            trend_features = self._extract_trend_features(df_with_indicators)
            features.extend(trend_features)
            
            # 2. 动量特征
            momentum_features = self._extract_momentum_features(df_with_indicators)
            features.extend(momentum_features)
            
            # 3. 波动率特征
            volatility_features = self._extract_volatility_features(df_with_indicators)
            features.extend(volatility_features)
            
            # 4. 成交量特征
            volume_features = self._extract_volume_features(df_with_indicators)
            features.extend(volume_features)
            
            # 5. 市场结构特征
            structure_features = self._extract_structure_features(df_with_indicators)
            features.extend(structure_features)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return np.array([])
    
    def _extract_trend_features(self, df: pd.DataFrame) -> List[float]:
        """提取趋势特征（动态阈值）"""
        features = []
        close = df['close'].dropna()
        
        if len(close) < 100:
            return [0.0] * 15
        
        try:
            # 动态窗口价格变化（基于历史分位数）
            lookback_periods = [int(len(close) * 0.1), int(len(close) * 0.25)]  # 10%和25%历史长度
            for period in lookback_periods:
                if period > 0:
                    price_change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
                    # 使用历史分位数标准化
                    historical_changes = [(close.iloc[i] - close.iloc[i-period]) / close.iloc[i-period] 
                                        for i in range(period, len(close)-1)]
                    if historical_changes:
                        percentile = stats.percentileofscore(historical_changes, price_change) / 100
                        features.append(percentile)
                    else:
                        features.append(0.5)
                else:
                    features.append(0.5)
            
            # 移动平均相对强度（滚动Z-score）
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma_20 = df['sma_20'].dropna()
                sma_50 = df['sma_50'].dropna()
                
                if len(sma_20) >= 50 and len(sma_50) >= 50:
                    # MA斜率的历史分位数
                    ma20_slopes = [(sma_20.iloc[i] - sma_20.iloc[i-10]) / sma_20.iloc[i-10] 
                                  for i in range(10, len(sma_20)-1)]
                    current_ma20_slope = (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10]
                    ma20_slope_percentile = stats.percentileofscore(ma20_slopes, current_ma20_slope) / 100 if ma20_slopes else 0.5
                    
                    # 价格相对MA位置的Z-score
                    price_ma_ratios = close / sma_20
                    current_ratio = price_ma_ratios.iloc[-1]
                    ratio_zscore = (current_ratio - price_ma_ratios.rolling(50).mean().iloc[-1]) / price_ma_ratios.rolling(50).std().iloc[-1]
                    ratio_zscore = np.tanh(ratio_zscore / 2) if not np.isnan(ratio_zscore) else 0  # 标准化到[-1,1]
                    
                    # MA交叉强度
                    ma_cross_ratio = sma_20.iloc[-1] / sma_50.iloc[-1] - 1
                    historical_cross = [sma_20.iloc[i] / sma_50.iloc[i] - 1 for i in range(len(sma_50)-50, len(sma_50))]
                    cross_percentile = stats.percentileofscore(historical_cross, ma_cross_ratio) / 100 if historical_cross else 0.5
                    
                    features.extend([ma20_slope_percentile, ratio_zscore, cross_percentile])
                else:
                    features.extend([0.5, 0.0, 0.5])
            else:
                features.extend([0.5, 0.0, 0.5])
            
            # ADX相对强度
            if 'adx' in df.columns:
                adx = df['adx'].dropna()
                if len(adx) >= 50:
                    adx_percentile = stats.percentileofscore(adx.iloc[-50:], adx.iloc[-1]) / 100
                    features.append(adx_percentile)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # MACD动量持续性
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                macd = df['macd'].dropna()
                macd_signal = df['macd_signal'].dropna()
                
                if len(macd) >= 50:
                    macd_hist = macd - macd_signal
                    current_hist = macd_hist.iloc[-1]
                    hist_percentile = stats.percentileofscore(macd_hist.iloc[-50:], current_hist) / 100
                    
                    # MACD方向一致性（最近N期同向比例）
                    recent_directions = np.sign(macd_hist.iloc[-10:])
                    direction_consistency = abs(recent_directions.mean())
                    
                    features.extend([hist_percentile, direction_consistency])
                else:
                    features.extend([0.5, 0.5])
            else:
                features.extend([0.5, 0.5])
            
            # 动态布林带位置
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                bb_upper = df['bb_upper'].dropna()
                bb_lower = df['bb_lower'].dropna()
                
                if len(bb_upper) >= 50:
                    bb_positions = [(close.iloc[i] - bb_lower.iloc[i]) / (bb_upper.iloc[i] - bb_lower.iloc[i]) 
                                   for i in range(len(bb_upper)-50, len(bb_upper))]
                    current_position = bb_positions[-1] if bb_positions else 0.5
                    position_percentile = stats.percentileofscore(bb_positions, current_position) / 100
                    
                    bb_widths = [(bb_upper.iloc[i] - bb_lower.iloc[i]) / close.iloc[i] for i in range(len(bb_upper)-50, len(bb_upper))]
                    current_width = bb_widths[-1] if bb_widths else 0.1
                    width_percentile = stats.percentileofscore(bb_widths, current_width) / 100
                    
                    features.extend([position_percentile, width_percentile])
                else:
                    features.extend([0.5, 0.5])
            else:
                features.extend([0.5, 0.5])
            
            # 动态支撑阻力突破强度
            window = min(50, len(close) // 4)
            if window >= 10:
                rolling_highs = close.rolling(window).max()
                rolling_lows = close.rolling(window).min()
                
                # 突破强度（相对于历史突破幅度）
                breakout_strength = 0
                if close.iloc[-1] > rolling_highs.iloc[-2]:
                    breakout_pct = (close.iloc[-1] - rolling_highs.iloc[-2]) / rolling_highs.iloc[-2]
                    historical_breakouts = [max(0, (close.iloc[i] - rolling_highs.iloc[i-1]) / rolling_highs.iloc[i-1]) 
                                          for i in range(window, len(close)-1) if close.iloc[i] > rolling_highs.iloc[i-1]]
                    if historical_breakouts:
                        breakout_strength = stats.percentileofscore(historical_breakouts, breakout_pct) / 100
                
                breakdown_strength = 0
                if close.iloc[-1] < rolling_lows.iloc[-2]:
                    breakdown_pct = (rolling_lows.iloc[-2] - close.iloc[-1]) / rolling_lows.iloc[-2]
                    historical_breakdowns = [max(0, (rolling_lows.iloc[i-1] - close.iloc[i]) / rolling_lows.iloc[i-1]) 
                                           for i in range(window, len(close)-1) if close.iloc[i] < rolling_lows.iloc[i-1]]
                    if historical_breakdowns:
                        breakdown_strength = stats.percentileofscore(historical_breakdowns, breakdown_pct) / 100
                
                features.extend([breakout_strength, breakdown_strength])
            else:
                features.extend([0.0, 0.0])
            
        except Exception as e:
            self.logger.error(f"趋势特征提取失败: {e}")
            features.extend([0.0] * (15 - len(features)))
        
        return features[:15]
    
    def _extract_momentum_features(self, df: pd.DataFrame) -> List[float]:
        """提取动量特征（自适应阈值）"""
        features = []
        
        try:
            # 动态RSI分析
            if 'rsi' in df.columns:
                rsi = df['rsi'].dropna()
                if len(rsi) >= 50:
                    current_rsi = rsi.iloc[-1]
                    rsi_percentile = stats.percentileofscore(rsi.iloc[-50:], current_rsi) / 100
                    
                    # 动态超买超卖线（基于历史分位数）
                    rsi_25th = np.percentile(rsi.iloc[-100:] if len(rsi) >= 100 else rsi, 25)
                    rsi_75th = np.percentile(rsi.iloc[-100:] if len(rsi) >= 100 else rsi, 75)
                    
                    adaptive_oversold = 1 if current_rsi < rsi_25th else 0
                    adaptive_overbought = 1 if current_rsi > rsi_75th else 0
                    
                    features.extend([rsi_percentile, adaptive_oversold, adaptive_overbought])
                else:
                    features.extend([0.5, 0, 0])
            else:
                features.extend([0.5, 0, 0])
            
            # 随机指标相对强度
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                stoch_k = df['stoch_k'].dropna()
                stoch_d = df['stoch_d'].dropna()
                
                if len(stoch_k) >= 50:
                    # 当前位置的历史分位数
                    stoch_percentile = stats.percentileofscore(stoch_k.iloc[-50:], stoch_k.iloc[-1]) / 100
                    
                    # K-D差值的相对强度
                    kd_diff = stoch_k - stoch_d
                    current_diff = kd_diff.iloc[-1]
                    diff_percentile = stats.percentileofscore(kd_diff.iloc[-50:], current_diff) / 100
                    
                    features.extend([stoch_percentile, diff_percentile])
                else:
                    features.extend([0.5, 0.5])
            else:
                features.extend([0.5, 0.5])
            
            # 威廉指标自适应
            close = df['close'].dropna()
            high = df['high'].dropna()
            low = df['low'].dropna()
            
            if len(close) >= 50:
                williams_r = self.technical_indicators.calculate_williams_r(high.values, low.values, close.values)
                if len(williams_r) >= 50:
                    current_wr = williams_r[-1]
                    wr_percentile = stats.percentileofscore(williams_r[-50:], current_wr) / 100
                    features.append(wr_percentile)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 动量持续性分析
            if len(close) >= 30:
                momentum_10 = self.technical_indicators.calculate_momentum(close.values, 10)
                momentum_20 = self.technical_indicators.calculate_momentum(close.values, 20)
                
                if len(momentum_10) >= 20 and len(momentum_20) >= 20:
                    # 多时间框架动量一致性
                    mom10_recent = momentum_10[-10:]
                    mom20_recent = momentum_20[-10:]
                    
                    # 动量方向一致性
                    mom10_direction = np.mean(np.sign(mom10_recent))
                    mom20_direction = np.mean(np.sign(mom20_recent))
                    momentum_consensus = (mom10_direction + mom20_direction) / 2
                    
                    features.append(momentum_consensus)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # ROC多时间框架分析
            if len(close) >= 60:
                roc_short = self.technical_indicators.calculate_roc(close.values, 5)
                roc_medium = self.technical_indicators.calculate_roc(close.values, 12)
                roc_long = self.technical_indicators.calculate_roc(close.values, 25)
                
                if all(len(roc) >= 30 for roc in [roc_short, roc_medium, roc_long]):
                    # ROC趋势一致性评分
                    roc_directions = [np.sign(roc[-1]) for roc in [roc_short, roc_medium, roc_long] if not np.isnan(roc[-1])]
                    if roc_directions:
                        roc_consistency = abs(np.mean(roc_directions))
                        features.append(roc_consistency)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # CCI自适应阈值
            if len(close) >= 50:
                cci = self.technical_indicators.calculate_cci(high.values, low.values, close.values)
                if len(cci) >= 50:
                    current_cci = cci[-1]
                    # 使用历史CCI分布确定当前位置
                    cci_percentile = stats.percentileofscore(cci[-100:] if len(cci) >= 100 else cci[-50:], current_cci) / 100
                    features.append(cci_percentile)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
        except Exception as e:
            self.logger.error(f"动量特征提取失败: {e}")
            features.extend([0.0] * (10 - len(features)))
        
        return features[:10]
    
    def _extract_volatility_features(self, df: pd.DataFrame) -> List[float]:
        """提取波动率特征（动态基准）"""
        features = []
        close = df['close'].dropna()
        
        try:
            if len(close) < 60:
                return [0.0] * 8
            
            returns = close.pct_change().dropna()
            
            # 多时间框架波动率比较
            lookback_windows = [5, 20, 60]
            volatilities = []
            
            for window in lookback_windows:
                if len(returns) >= window:
                    vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
                    volatilities.append(vol)
            
            if len(volatilities) >= 2:
                # 短期vs长期波动率比率的历史分位数
                short_long_ratio = volatilities[0] / volatilities[1] if volatilities[1] != 0 else 1
                
                # 计算历史比率分布
                historical_ratios = []
                for i in range(60, len(returns)):
                    vol_5 = returns.iloc[i-5:i].std() * np.sqrt(252)
                    vol_20 = returns.iloc[i-20:i].std() * np.sqrt(252)
                    if vol_20 != 0:
                        historical_ratios.append(vol_5 / vol_20)
                
                if historical_ratios:
                    vol_ratio_percentile = stats.percentileofscore(historical_ratios, short_long_ratio) / 100
                    features.append(vol_ratio_percentile)
                else:
                    features.append(0.5)
                
                # 当前波动率在历史分布中的位置
                current_vol = volatilities[1]  # 20日波动率
                historical_vols = [returns.iloc[i-20:i].std() * np.sqrt(252) for i in range(20, len(returns))]
                vol_percentile = stats.percentileofscore(historical_vols, current_vol) / 100
                features.append(vol_percentile)
            else:
                features.extend([0.5, 0.5])
            
            # 波动率趋势分析
            if len(volatilities) == 3:
                # 波动率斜率（短期到长期的趋势）
                vol_slope = (volatilities[0] - volatilities[2]) / volatilities[2] if volatilities[2] != 0 else 0
                vol_slope_norm = np.tanh(vol_slope)  # 标准化到[-1,1]
                features.append(vol_slope_norm)
            else:
                features.append(0.0)
            
            # 动态ATR分析
            if 'atr' in df.columns:
                atr = df['atr'].dropna()
                if len(atr) >= 50:
                    atr_pct = atr / close
                    current_atr_pct = atr_pct.iloc[-1]
                    
                    # ATR百分比的历史分位数
                    atr_percentile = stats.percentileofscore(atr_pct.iloc[-50:], current_atr_pct) / 100
                    features.append(atr_percentile)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 动态布林带宽度分析
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                bb_upper = df['bb_upper'].dropna()
                bb_lower = df['bb_lower'].dropna()
                
                if len(bb_upper) >= 50:
                    bb_widths = (bb_upper - bb_lower) / close
                    current_width = bb_widths.iloc[-1]
                    
                    # BB宽度的历史分位数
                    width_percentile = stats.percentileofscore(bb_widths.iloc[-100:] if len(bb_widths) >= 100 else bb_widths.iloc[-50:], current_width) / 100
                    features.append(width_percentile)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 高低价差的相对强度
            high = df['high'].dropna()
            low = df['low'].dropna()
            
            if len(high) >= 50 and len(low) >= 50:
                hl_ratios = (high - low) / close
                current_hl_ratio = hl_ratios.iloc[-1]
                
                # 当日波动幅度在历史分布中的位置
                hl_percentile = stats.percentileofscore(hl_ratios.iloc[-50:], current_hl_ratio) / 100
                
                # 波动幅度趋势（最近10日平均vs历史平均）
                recent_avg = hl_ratios.iloc[-10:].mean()
                historical_avg = hl_ratios.iloc[-50:-10].mean() if len(hl_ratios) >= 60 else hl_ratios.mean()
                hl_trend = recent_avg / historical_avg - 1 if historical_avg != 0 else 0
                hl_trend_norm = np.tanh(hl_trend * 5)  # 标准化
                
                features.extend([hl_percentile, hl_trend_norm])
            else:
                features.extend([0.5, 0.0])
            
            # 动态GARCH波动率预测
            if len(returns) >= 100:
                garch_vol = self._adaptive_garch_forecast(returns.values)
                current_vol = returns.iloc[-20:].std() * np.sqrt(252)
                
                # GARCH预测vs当前波动率
                garch_ratio = garch_vol / current_vol if current_vol != 0 else 1
                garch_signal = np.tanh((garch_ratio - 1) * 2)  # 标准化预测信号
                features.append(garch_signal)
            else:
                features.append(0.0)
            
        except Exception as e:
            self.logger.error(f"波动率特征提取失败: {e}")
            features.extend([0.0] * (8 - len(features)))
        
        return features[:8]
    
    def _adaptive_garch_forecast(self, returns: np.ndarray) -> float:
        """自适应GARCH波动率预测"""
        if len(returns) < 50:
            return np.std(returns) * np.sqrt(252)
        
        # 动态参数估计
        recent_returns = returns[-100:]
        
        # 自适应alpha和beta参数
        squared_returns = recent_returns ** 2
        mean_sq_return = np.mean(squared_returns)
        
        # 简化的参数估计
        alpha = min(0.3, np.std(squared_returns) / mean_sq_return) if mean_sq_return != 0 else 0.1
        beta = max(0.5, 1 - 2 * alpha)
        
        # GARCH预测
        variance = np.var(recent_returns)
        for i in range(1, min(len(recent_returns), 30)):
            variance = (1 - alpha - beta) * mean_sq_return + alpha * recent_returns[i-1]**2 + beta * variance
        
        return np.sqrt(variance * 252)
    
    def _extract_volume_features(self, df: pd.DataFrame) -> List[float]:
        """提取成交量特征"""
        features = []
        
        try:
            if 'volume' not in df.columns:
                return [0.0] * 7
            
            volume = df['volume'].dropna()
            close = df['close'].dropna()
            
            if len(volume) < 20 or len(close) < 20:
                return [0.0] * 7
            
            # 成交量比率
            vol_ratio_5 = volume.iloc[-1] / volume.rolling(5).mean().iloc[-1] if volume.rolling(5).mean().iloc[-1] != 0 else 1
            vol_ratio_20 = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] != 0 else 1
            features.extend([vol_ratio_5, vol_ratio_20])
            
            # VWAP相对位置
            if 'vwap' in df.columns:
                vwap = df['vwap'].dropna()
                if len(vwap) > 0:
                    vwap_ratio = close.iloc[-1] / vwap.iloc[-1] - 1
                    features.append(vwap_ratio)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # OBV趋势
            if 'obv' in df.columns:
                obv = df['obv'].dropna()
                if len(obv) >= 10:
                    obv_trend = (obv.iloc[-1] - obv.iloc[-10]) / obv.iloc[-10] if obv.iloc[-10] != 0 else 0
                    features.append(obv_trend)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # 资金流量指标
            if 'mfi' in df.columns:
                mfi = df['mfi'].dropna()
                if len(mfi) > 0:
                    mfi_norm = mfi.iloc[-1] / 100
                    features.append(mfi_norm)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 成交量加权收益率
            if len(volume) >= 5:
                returns = close.pct_change().dropna()
                if len(returns) >= 5:
                    vol_weighted_return = (returns.iloc[-5:] * volume.iloc[-5:]).sum() / volume.iloc[-5:].sum()
                    features.append(vol_weighted_return)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # 成交量标准差
            vol_std = volume.rolling(20).std().iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] != 0 else 0
            features.append(vol_std)
            
        except Exception as e:
            self.logger.error(f"成交量特征提取失败: {e}")
            features.extend([0.0] * (7 - len(features)))
        
        return features[:7]
    
    def _extract_structure_features(self, df: pd.DataFrame) -> List[float]:
        """提取市场结构特征"""
        features = []
        close = df['close'].dropna()
        high = df['high'].dropna()
        low = df['low'].dropna()
        
        try:
            if len(close) < 50:
                return [0.0] * 10
            
            # 分形维数（简化计算）
            fractal_dim = self._calculate_fractal_dimension(close.values[-50:])
            features.append(fractal_dim)
            
            # 赫斯特指数
            hurst = self._calculate_hurst_exponent(close.values[-100:] if len(close) >= 100 else close.values)
            features.append(hurst)
            
            # 支撑阻力强度
            support_strength, resistance_strength = self._calculate_support_resistance_strength(high.values, low.values, close.values)
            features.extend([support_strength, resistance_strength])
            
            # 价格缺口检测
            gaps = self._detect_price_gaps(df)
            features.append(gaps)
            
            # 市场效率比率
            efficiency_ratio = self._calculate_efficiency_ratio(close.values[-20:] if len(close) >= 20 else close.values)
            features.append(efficiency_ratio)
            
            # 均值回归强度
            mean_reversion = self._calculate_mean_reversion_strength(close.values)
            features.append(mean_reversion)
            
            # 长期记忆性
            long_memory = self._calculate_long_memory(close.values)
            features.append(long_memory)
            
            # 非线性相关性
            nonlinear_corr = self._calculate_nonlinear_correlation(close.values)
            features.append(nonlinear_corr)
            
            # 市场微观结构噪音
            microstructure_noise = self._calculate_microstructure_noise(close.values)
            features.append(microstructure_noise)
            
        except Exception as e:
            self.logger.error(f"结构特征提取失败: {e}")
            features.extend([0.0] * (10 - len(features)))
        
        return features[:10]
    
    def _simple_garch_forecast(self, returns: np.ndarray, alpha: float = 0.1, beta: float = 0.85) -> float:
        """简化GARCH波动率预测"""
        if len(returns) < 10:
            return np.std(returns) * np.sqrt(252)
        
        # 简化的GARCH(1,1)模型
        variance = np.var(returns)
        for i in range(1, min(len(returns), 50)):
            variance = (1 - alpha - beta) * np.var(returns) + alpha * returns[i-1]**2 + beta * variance
        
        return np.sqrt(variance * 252)
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """计算分形维数"""
        try:
            n = len(prices)
            if n < 10:
                return 1.5
            
            # 使用Higuchi方法
            k_max = min(10, n // 4)
            lk = []
            
            for k in range(1, k_max + 1):
                lm = []
                for m in range(k):
                    lm_k = 0
                    for i in range(1, (n - m) // k):
                        lm_k += abs(prices[m + i * k] - prices[m + (i - 1) * k])
                    lm_k = lm_k * (n - 1) / (k * k * ((n - m) // k))
                    lm.append(lm_k)
                lk.append(np.mean(lm))
            
            # 线性回归求斜率
            log_k = np.log(range(1, k_max + 1))
            log_lk = np.log(lk)
            slope = np.polyfit(log_k, log_lk, 1)[0]
            
            return 2 - slope
            
        except:
            return 1.5
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """计算赫斯特指数"""
        try:
            if len(prices) < 20:
                return 0.5
            
            # R/S方法
            returns = np.diff(np.log(prices))
            n = len(returns)
            
            # 不同时间窗口
            lags = range(2, min(n // 4, 20))
            rs = []
            
            for lag in lags:
                chunks = n // lag
                rs_values = []
                
                for i in range(chunks):
                    chunk = returns[i * lag:(i + 1) * lag]
                    if len(chunk) == lag:
                        mean_return = np.mean(chunk)
                        deviations = np.cumsum(chunk - mean_return)
                        R = np.max(deviations) - np.min(deviations)
                        S = np.std(chunk)
                        if S > 0:
                            rs_values.append(R / S)
                
                if rs_values:
                    rs.append(np.mean(rs_values))
            
            if len(rs) > 1:
                log_lags = np.log(lags[:len(rs)])
                log_rs = np.log(rs)
                hurst = np.polyfit(log_lags, log_rs, 1)[0]
                return max(0, min(1, hurst))
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_support_resistance_strength(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float]:
        """计算支撑阻力强度"""
        try:
            if len(close) < 20:
                return 0.5, 0.5
            
            # 找到局部高低点
            high_peaks = argrelextrema(high, np.greater, order=5)[0]
            low_peaks = argrelextrema(low, np.less, order=5)[0]
            
            current_price = close[-1]
            
            # 计算阻力强度
            resistance_levels = high[high_peaks[high_peaks >= len(high) - 50]] if len(high_peaks) > 0 else []
            resistance_strength = 0
            for level in resistance_levels:
                if level > current_price:
                    distance = abs(level - current_price) / current_price
                    if distance < 0.05:  # 5%以内
                        resistance_strength += 1 / (1 + distance * 10)
            
            # 计算支撑强度
            support_levels = low[low_peaks[low_peaks >= len(low) - 50]] if len(low_peaks) > 0 else []
            support_strength = 0
            for level in support_levels:
                if level < current_price:
                    distance = abs(current_price - level) / current_price
                    if distance < 0.05:  # 5%以内
                        support_strength += 1 / (1 + distance * 10)
            
            # 标准化
            resistance_strength = min(1, resistance_strength / 5)
            support_strength = min(1, support_strength / 5)
            
            return support_strength, resistance_strength
            
        except:
            return 0.5, 0.5
    
    def _detect_price_gaps(self, df: pd.DataFrame) -> float:
        """检测价格缺口"""
        try:
            if len(df) < 10:
                return 0
            
            gaps = 0
            for i in range(1, min(len(df), 20)):
                prev_high = df['high'].iloc[i-1]
                prev_low = df['low'].iloc[i-1]
                curr_high = df['high'].iloc[i]
                curr_low = df['low'].iloc[i]
                
                # 向上缺口
                if curr_low > prev_high:
                    gap_size = (curr_low - prev_high) / prev_high
                    if gap_size > 0.001:  # 0.1%以上
                        gaps += gap_size
                
                # 向下缺口
                elif curr_high < prev_low:
                    gap_size = (prev_low - curr_high) / prev_low
                    if gap_size > 0.001:  # 0.1%以上
                        gaps += gap_size
            
            return min(1, gaps * 100)
            
        except:
            return 0
    
    def _calculate_efficiency_ratio(self, prices: np.ndarray) -> float:
        """计算市场效率比率"""
        try:
            if len(prices) < 3:
                return 0.5
            
            # 净变化 / 总变化
            net_change = abs(prices[-1] - prices[0])
            total_change = np.sum(np.abs(np.diff(prices)))
            
            if total_change == 0:
                return 0
            
            efficiency = net_change / total_change
            return min(1, efficiency)
            
        except:
            return 0.5
    
    def _calculate_mean_reversion_strength(self, prices: np.ndarray) -> float:
        """计算均值回归强度"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # 计算价格偏离均值的程度和回归速度
            mean_price = np.mean(prices)
            deviations = (prices - mean_price) / mean_price
            
            # 计算自相关系数
            if len(deviations) > 1:
                correlation = np.corrcoef(deviations[:-1], deviations[1:])[0, 1]
                # 负相关表示均值回归
                mean_reversion = max(0, -correlation)
                return min(1, mean_reversion)
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_long_memory(self, prices: np.ndarray) -> float:
        """计算长期记忆性"""
        try:
            if len(prices) < 20:
                return 0.5
            
            returns = np.diff(np.log(prices))
            
            # 计算不同滞后期的自相关
            max_lag = min(10, len(returns) // 4)
            autocorrs = []
            
            for lag in range(1, max_lag + 1):
                if len(returns) > lag:
                    corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))
            
            if autocorrs:
                # 长期记忆通过自相关的衰减速度衡量
                return min(1, np.mean(autocorrs))
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_nonlinear_correlation(self, prices: np.ndarray) -> float:
        """计算非线性相关性"""
        try:
            if len(prices) < 10:
                return 0.5
            
            returns = np.diff(np.log(prices))
            
            # 计算收益率平方的自相关（ARCH效应）
            squared_returns = returns ** 2
            
            if len(squared_returns) > 1:
                corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                if not np.isnan(corr):
                    return min(1, abs(corr))
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_microstructure_noise(self, prices: np.ndarray) -> float:
        """计算微观结构噪音"""
        try:
            if len(prices) < 5:
                return 0.5
            
            # 通过价格变化的波动率衡量噪音
            price_changes = np.diff(prices)
            relative_changes = price_changes / prices[:-1]
            
            # 计算变化率的标准差
            noise_level = np.std(relative_changes)
            
            # 标准化
            return min(1, noise_level * 100)
            
        except:
            return 0.5
    
    def train_classifier(self, training_data: List[pd.DataFrame], labels: List[str] = None):
        """训练市场环境分类器"""
        try:
            self.logger.info("开始训练市场环境分类器...")
            
            # 提取特征
            features_list = []
            for df in training_data:
                features = self.extract_market_features(df)
                if len(features) > 0:
                    features_list.append(features)
            
            if not features_list:
                self.logger.error("没有有效的特征数据")
                return False
            
            X = np.array(features_list)
            
            # 特征标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 降维
            X_pca = self.pca.fit_transform(X_scaled)
            
            # 无监督聚类（如果没有标签）
            if labels is None:
                self.classifier.fit(X_pca)
            else:
                # 如果有标签，可以使用监督学习
                from sklearn.ensemble import RandomForestClassifier
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # 将标签映射为数字
                label_mapping = {state.value: i for i, state in enumerate(MarketState)}
                y = np.array([label_mapping.get(label, 0) for label in labels])
                
                self.classifier.fit(X_pca, y)
            
            # 保存模型
            self.save_model()
            
            self.logger.info("市场环境分类器训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"分类器训练失败: {e}")
            return False
    
    def classify_market_environment(self, df: pd.DataFrame) -> Tuple[str, float]:
        """分类市场环境"""
        try:
            # 提取特征
            features = self.extract_market_features(df)
            
            if len(features) == 0:
                return MarketState.SIDEWAYS, 0.5
            
            # 预处理
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            
            # 预测
            prediction = self.classifier.predict(X_pca)[0]
            
            # 计算置信度
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X_pca)[0]
                confidence = np.max(probabilities)
            elif hasattr(self.classifier, 'transform'):
                # KMeans情况下，用距离计算置信度
                distances = self.classifier.transform(X_pca)[0]
                min_distance = np.min(distances)
                confidence = 1 / (1 + min_distance)
            else:
                confidence = 0.7
            
            # 映射到市场状态
            market_state = self.state_mapping.get(prediction, MarketState.SIDEWAYS)
            
            return market_state, confidence
            
        except Exception as e:
            self.logger.error(f"市场环境分类失败: {e}")
            return MarketState.SIDEWAYS, 0.5
    
    def get_market_regime_features(self, symbol: str) -> Dict[str, Any]:
        """获取市场制度特征"""
        try:
            # 从数据库获取历史数据
            from database_manager import DatabaseManager
            db = DatabaseManager()
            
            end_time = TimeUtils.timestamp_to_datetime(TimeUtils.now_timestamp())
            start_time = end_time - pd.Timedelta(days=30)
            
            df = db.db.query_market_data(symbol, start_time, end_time)
            
            if df.empty:
                return {}
            
            # 分类当前市场环境
            current_state, confidence = self.classify_market_environment(df)
            
            # 计算市场制度稳定性
            regime_stability = self._calculate_regime_stability(df)
            
            # 预测市场制度转换概率
            transition_prob = self._predict_regime_transition(df)
            
            return {
                'current_state': current_state,
                'confidence': confidence,
                'regime_stability': regime_stability,
                'transition_probability': transition_prob,
                'market_stress_level': self._calculate_market_stress(df),
                'liquidity_conditions': self._assess_liquidity_conditions(df)
            }
            
        except Exception as e:
            self.logger.error(f"获取市场制度特征失败: {e}")
            return {}
    
    def _calculate_regime_stability(self, df: pd.DataFrame) -> float:
        """计算市场制度稳定性"""
        try:
            # 滚动窗口分类
            window_size = 20
            classifications = []
            
            for i in range(window_size, len(df)):
                window_df = df.iloc[i-window_size:i]
                state, _ = self.classify_market_environment(window_df)
                classifications.append(state)
            
            if not classifications:
                return 0.5
            
            # 计算状态变化频率
            changes = sum(1 for i in range(1, len(classifications)) 
                         if classifications[i] != classifications[i-1])
            
            stability = 1 - (changes / len(classifications))
            return max(0, min(1, stability))
            
        except:
            return 0.5
    
    def _predict_regime_transition(self, df: pd.DataFrame) -> Dict[str, float]:
        """预测市场制度转换概率"""
        try:
            current_features = self.extract_market_features(df)
            
            # 简化的转换概率估计
            # 基于特征的极值程度
            feature_extremes = []
            for i, feature in enumerate(current_features):
                if abs(feature) > 2:  # 标准差的2倍以上
                    feature_extremes.append(abs(feature))
            
            if feature_extremes:
                avg_extreme = np.mean(feature_extremes)
                transition_prob = min(0.8, avg_extreme / 3)
            else:
                transition_prob = 0.1
            
            return {
                'bull_to_bear': transition_prob if current_features[0] < -1 else 0.1,
                'bear_to_bull': transition_prob if current_features[0] > 1 else 0.1,
                'trend_to_sideways': transition_prob if abs(current_features[1]) > 1.5 else 0.2,
                'low_to_high_vol': transition_prob if current_features[20] < -1 else 0.15,
                'high_to_low_vol': transition_prob if current_features[20] > 1 else 0.15
            }
            
        except:
            return {'bull_to_bear': 0.1, 'bear_to_bull': 0.1, 'trend_to_sideways': 0.2,
                   'low_to_high_vol': 0.15, 'high_to_low_vol': 0.15}
    
    def _calculate_market_stress(self, df: pd.DataFrame) -> float:
        """计算市场压力水平"""
        try:
            if len(df) < 20:
                return 0.5
            
            close = df['close'].dropna()
            volume = df['volume'].dropna() if 'volume' in df.columns else None
            
            stress_indicators = []
            
            # 价格波动率压力
            returns = close.pct_change().dropna()
            vol_stress = np.std(returns.iloc[-20:]) * np.sqrt(252)
            stress_indicators.append(min(1, vol_stress * 5))
            
            # 成交量异常
            if volume is not None and len(volume) >= 20:
                vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
                vol_stress = min(1, max(0, (vol_ratio - 1) / 2))
                stress_indicators.append(vol_stress)
            
            # 价格跳跃
            price_jumps = abs(returns.iloc[-10:])
            jump_stress = np.mean(price_jumps > 0.02)  # 2%以上的跳跃
            stress_indicators.append(jump_stress)
            
            return np.mean(stress_indicators) if stress_indicators else 0.5
            
        except:
            return 0.5
    
    def _assess_liquidity_conditions(self, df: pd.DataFrame) -> float:
        """评估流动性条件"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.5
            
            volume = df['volume'].dropna()
            close = df['close'].dropna()
            
            # 成交量一致性
            vol_cv = np.std(volume.iloc[-20:]) / np.mean(volume.iloc[-20:])
            vol_consistency = 1 / (1 + vol_cv)
            
            # 价量关系
            returns = close.pct_change().dropna()
            if len(returns) >= 20 and len(volume) >= 20:
                price_volume_corr = abs(np.corrcoef(returns.iloc[-20:], volume.iloc[-20:])[0, 1])
                if np.isnan(price_volume_corr):
                    price_volume_corr = 0
            else:
                price_volume_corr = 0
            
            # 流动性评分
            liquidity_score = (vol_consistency + price_volume_corr) / 2
            return max(0, min(1, liquidity_score))
            
        except:
            return 0.5
    
    def save_model(self):
        """保存模型"""
        try:
            model_data = {
                'scaler': self.scaler,
                'pca': self.pca,
                'classifier': self.classifier,
                'state_mapping': self.state_mapping
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.logger.info("市场环境分类器模型已保存")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def load_model(self):
        """加载模型"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.scaler = model_data['scaler']
                self.pca = model_data['pca']
                self.classifier = model_data['classifier']
                self.state_mapping = model_data['state_mapping']
                
                self.logger.info("市场环境分类器模型已加载")
                return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
        
        return False


# 测试函数
async def test_technical_analysis():
    """测试技术分析模块"""
    logger = TradingLogger().get_logger("TECHNICAL_ANALYSIS_TEST")
    logger.info("🧪 开始技术分析测试...")
    
    # 测试技术指标计算
    indicators = TechnicalIndicators()
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    test_df = pd.DataFrame({
        'timestamp': [int(d.timestamp()) for d in dates],
        'open': prices + np.random.randn(200) * 50,
        'high': prices + abs(np.random.randn(200) * 100),
        'low': prices - abs(np.random.randn(200) * 100),
        'close': prices,
        'volume': np.random.exponential(1000, 200)
    })
    
    # 计算所有指标
    logger.info("计算技术指标...")
    df_with_indicators = indicators.get_all_indicators(test_df)
    logger.info(f"计算了 {len(df_with_indicators.columns) - len(test_df.columns)} 个技术指标")
    
    # 测试实时指标计算
    indicators.init_symbol_buffers("BTCUSDT")
    for _, row in test_df.iterrows():
        indicators.update_market_data("BTCUSDT", row.to_dict())
    
    realtime_indicators = indicators.calculate_realtime_indicators("BTCUSDT")
    logger.info(f"实时指标: {len(realtime_indicators)} 个")
    
    # 测试市场环境分类
    logger.info("测试市场环境分类...")
    classifier = MarketEnvironmentClassifier()
    
    market_state, confidence = classifier.classify_market_environment(test_df)
    logger.info(f"市场状态: {market_state}, 置信度: {confidence:.2f}")
    
    # 获取市场制度特征
    regime_features = classifier.get_market_regime_features("BTCUSDT")
    logger.info(f"市场制度特征: {regime_features}")
    
    logger.info("✅ 技术分析测试完成")


