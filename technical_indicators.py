# =============================================================================
# STARK4-ML 技术分析层 - 技术指标计算与市场环境识别
# 依赖: logger.py, utils.py, database_manager.py
# =============================================================================

# technical_indicators.py - 技术指标计算
# =============================================================================

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import deque
from scipy.signal import argrelextrema
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from logger import TradingLogger
from utils import ConfigManager, TimeUtils


class RollingBuffer:
    """滚动缓冲区，用于实时计算"""
    
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
    
    def append(self, value):
        self.data.append(value)
    
    def extend(self, values):
        self.data.extend(values)
    
    def get_array(self) -> np.ndarray:
        return np.array(self.data)
    
    def is_full(self) -> bool:
        return len(self.data) == self.maxlen
    
    def __len__(self):
        return len(self.data)


class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("TECHNICAL_INDICATORS")
        
        # 实时计算缓冲区
        self.buffers: Dict[str, Dict[str, RollingBuffer]] = {}
        
        # 指标参数配置
        self.params = {
            'sma_periods': [5, 10, 20, 50, 100, 200],
            'ema_periods': [5, 12, 21, 26, 50, 100],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'cci_period': 20,
            'williams_r_period': 14,
            'adx_period': 14,
            'momentum_period': 10,
            'roc_period': 12,
            'ultimate_oscillator_periods': [7, 14, 28],
            'trix_period': 14,
            'dmi_period': 14,
            'ppo_fast': 12,
            'ppo_slow': 26,
            'ppo_signal': 9,
            'keltner_period': 20,
            'keltner_multiplier': 2,
            'donchian_period': 20,
            'ichimoku_conversion': 9,
            'ichimoku_base': 26,
            'ichimoku_lagging': 52,
            'fisher_period': 10,
            'gamma_period': 20
        }
        
        # 更新配置参数
        for key, default_value in self.params.items():
            self.params[key] = self.config.get(f"indicators.{key}", default_value)
    
    def init_symbol_buffers(self, symbol: str, buffer_size: int = 1000):
        """初始化交易对的数据缓冲区"""
        if symbol not in self.buffers:
            self.buffers[symbol] = {}
        
        # 价格数据缓冲区
        price_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in price_fields:
            self.buffers[symbol][field] = RollingBuffer(buffer_size)
        
        # 订单薄数据缓冲区
        orderbook_fields = ['bid_price', 'ask_price', 'bid_size', 'ask_size', 'spread']
        for field in orderbook_fields:
            self.buffers[symbol][field] = RollingBuffer(buffer_size)
        
        # 交易数据缓冲区
        trade_fields = ['trade_price', 'trade_size', 'trade_count', 'buy_ratio']
        for field in trade_fields:
            self.buffers[symbol][field] = RollingBuffer(buffer_size)
    
    def update_market_data(self, symbol: str, data: Dict[str, float]):
        """更新市场数据到缓冲区"""
        if symbol not in self.buffers:
            self.init_symbol_buffers(symbol)
        
        # 更新OHLCV数据
        if all(key in data for key in ['open', 'high', 'low', 'close', 'volume']):
            for key in ['open', 'high', 'low', 'close', 'volume']:
                self.buffers[symbol][key].append(data[key])
    
    def update_orderbook_data(self, symbol: str, data: Dict[str, float]):
        """更新订单薄数据"""
        if symbol not in self.buffers:
            self.init_symbol_buffers(symbol)
        
        # 计算衍生指标
        if 'bid_price' in data and 'ask_price' in data:
            data['spread'] = data['ask_price'] - data['bid_price']
        
        for key in ['bid_price', 'ask_price', 'bid_size', 'ask_size', 'spread']:
            if key in data:
                self.buffers[symbol][key].append(data[key])
    
    def update_trade_data(self, symbol: str, trades: List[Dict[str, Any]]):
        """更新交易数据"""
        if symbol not in self.buffers:
            self.init_symbol_buffers(symbol)
        
        if not trades:
            return
        
        # 聚合交易数据
        total_size = sum(t['quantity'] for t in trades)
        avg_price = sum(t['price'] * t['quantity'] for t in trades) / total_size if total_size > 0 else 0
        buy_count = sum(1 for t in trades if not t.get('is_buyer_maker', False))
        buy_ratio = buy_count / len(trades) if trades else 0.5
        
        self.buffers[symbol]['trade_price'].append(avg_price)
        self.buffers[symbol]['trade_size'].append(total_size)
        self.buffers[symbol]['trade_count'].append(len(trades))
        self.buffers[symbol]['buy_ratio'].append(buy_ratio)
    
    # ========== 基础移动平均指标 ==========
    
    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        return talib.SMA(prices, timeperiod=period)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """指数移动平均"""
        return talib.EMA(prices, timeperiod=period)
    
    def calculate_wma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """加权移动平均"""
        return talib.WMA(prices, timeperiod=period)
    
    def calculate_dema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """双指数移动平均"""
        return talib.DEMA(prices, timeperiod=period)
    
    def calculate_tema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """三指数移动平均"""
        return talib.TEMA(prices, timeperiod=period)
    
    def calculate_kama(self, prices: np.ndarray, period: int = 30) -> np.ndarray:
        """考夫曼自适应移动平均"""
        return talib.KAMA(prices, timeperiod=period)
    
    # ========== 动量指标 ==========
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """相对强弱指标"""
        return talib.RSI(prices, timeperiod=period)
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD指标"""
        macd, signal_line, histogram = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, signal_line, histogram
    
    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """随机指标"""
        k_percent = talib.STOCHK(high, low, close, fastk_period=k_period, slowk_period=3, slowk_matype=0)
        d_percent = talib.STOCHD(high, low, close, fastk_period=k_period, slowk_period=3, slowk_matype=0, slowd_period=d_period, slowd_matype=0)
        return k_percent, d_percent
    
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """商品通道指标"""
        return talib.CCI(high, low, close, timeperiod=period)
    
    def calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """威廉指标"""
        return talib.WILLR(high, low, close, timeperiod=period)
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """动量指标"""
        return talib.MOM(prices, timeperiod=period)
    
    def calculate_roc(self, prices: np.ndarray, period: int = 12) -> np.ndarray:
        """变化率指标"""
        return talib.ROC(prices, timeperiod=period)
    
    def calculate_ultimate_oscillator(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                    periods: List[int] = [7, 14, 28]) -> np.ndarray:
        """终极指标"""
        return talib.ULTOSC(high, low, close, timeperiod1=periods[0], timeperiod2=periods[1], timeperiod3=periods[2])
    
    def calculate_trix(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """TRIX指标"""
        return talib.TRIX(prices, timeperiod=period)
    
    # ========== 趋势指标 ==========
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """平均趋向指标"""
        return talib.ADX(high, low, close, timeperiod=period)
    
    def calculate_dmi(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """方向运动指标"""
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
        return plus_di, minus_di
    
    def calculate_aroon(self, high: np.ndarray, low: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """阿隆指标"""
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
        return aroon_up, aroon_down
    
    def calculate_ppo(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """价格震荡百分比"""
        ppo = talib.PPO(prices, fastperiod=fast, slowperiod=slow, matype=0)
        ppo_signal = talib.EMA(ppo, timeperiod=signal)
        return ppo, ppo_signal
    
    # ========== 波动率指标 ==========
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带"""
        upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
        return upper, middle, lower
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """平均真实波幅"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    def calculate_keltner_channels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                 period: int = 20, multiplier: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """肯特纳通道"""
        ema = talib.EMA(close, timeperiod=period)
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return upper, ema, lower
    
    def calculate_donchian_channels(self, high: np.ndarray, low: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """唐奇安通道"""
        upper = talib.MAX(high, timeperiod=period)
        lower = talib.MIN(low, timeperiod=period)
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    # ========== 成交量指标 ==========
    
    def calculate_vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """成交量加权平均价"""
        typical_price = (high + low + close) / 3
        return np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """能量潮指标"""
        return talib.OBV(close, volume)
    
    def calculate_ad_line(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """累积/派发线"""
        return talib.AD(high, low, close, volume)
    
    def calculate_chaikin_mf(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
        """蔡金资金流量"""
        return talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=period)
    
    def calculate_mfi(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """资金流量指标"""
        return talib.MFI(high, low, close, volume, timeperiod=period)
    
    def calculate_force_index(self, close: np.ndarray, volume: np.ndarray, period: int = 13) -> np.ndarray:
        """力度指标"""
        raw_force = (close - np.roll(close, 1)) * volume
        return talib.EMA(raw_force, timeperiod=period)
    
    def calculate_ease_of_movement(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """简易波动指标"""
        distance_moved = ((high + low) / 2) - ((np.roll(high, 1) + np.roll(low, 1)) / 2)
        box_height = volume / (high - low)
        emv = distance_moved / box_height
        return talib.SMA(emv, timeperiod=period)
    
    # ========== 高级指标 ==========
    
    def calculate_fisher_transform(self, high: np.ndarray, low: np.ndarray, period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """费雪变换"""
        hl2 = (high + low) / 2
        
        # 标准化到-1到1之间
        highest = talib.MAX(hl2, timeperiod=period)
        lowest = talib.MIN(hl2, timeperiod=period)
        
        raw_value = 2 * ((hl2 - lowest) / (highest - lowest) - 0.5)
        
        # 限制在-0.999到0.999之间
        raw_value = np.clip(raw_value, -0.999, 0.999)
        
        # 费雪变换
        fisher = np.zeros_like(raw_value)
        fisher[1:] = 0.5 * np.log((1 + raw_value[1:]) / (1 - raw_value[1:]))
        
        # 信号线（前一期的费雪变换值）
        fisher_signal = np.roll(fisher, 1)
        
        return fisher, fisher_signal
    
    def calculate_gamma(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Gamma指标（二阶导数）"""
        # 计算一阶导数（变化率）
        first_derivative = np.diff(close)
        
        # 计算二阶导数（加速度）
        second_derivative = np.diff(first_derivative)
        
        # 使用移动平均平滑
        gamma = np.zeros_like(close)
        gamma[2:] = second_derivative
        
        return talib.SMA(gamma, timeperiod=period)
    
    def calculate_schaff_trend_cycle(self, close: np.ndarray, fast: int = 23, slow: int = 50, cycle: int = 10) -> np.ndarray:
        """Schaff趋势周期"""
        # 计算MACD
        macd_line = talib.EMA(close, timeperiod=fast) - talib.EMA(close, timeperiod=slow)
        
        # 第一次随机化
        stoch1_k = talib.STOCHK(macd_line, macd_line, macd_line, fastk_period=cycle)
        stoch1_d = talib.SMA(stoch1_k, timeperiod=3)
        
        # 第二次随机化
        stoch2_k = talib.STOCHK(stoch1_d, stoch1_d, stoch1_d, fastk_period=cycle)
        stc = talib.SMA(stoch2_k, timeperiod=3)
        
        return stc
    
    def calculate_connors_rsi(self, close: np.ndarray, rsi_period: int = 3, 
                             updown_period: int = 2, roc_period: int = 100) -> np.ndarray:
        """康纳斯RSI"""
        # 组件1：RSI
        rsi = talib.RSI(close, timeperiod=rsi_period)
        
        # 组件2：连续上涨/下跌天数的RSI
        updown = np.zeros_like(close)
        streak = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                streak = max(1, streak + 1) if streak > 0 else 1
            elif close[i] < close[i-1]:
                streak = min(-1, streak - 1) if streak < 0 else -1
            else:
                streak = 0
            updown[i] = streak
        
        updown_rsi = talib.RSI(updown, timeperiod=updown_period)
        
        # 组件3：变化率百分位数
        roc = talib.ROC(close, timeperiod=1)
        roc_percentile = np.zeros_like(roc)
        
        for i in range(roc_period, len(roc)):
            window = roc[i-roc_period:i]
            percentile = (window < roc[i]).sum() / len(window) * 100
            roc_percentile[i] = percentile
        
        # 康纳斯RSI = (RSI + UpDown RSI + ROC percentile) / 3
        connors_rsi = (rsi + updown_rsi + roc_percentile) / 3
        
        return connors_rsi
    
    # ========== 订单薄指标 ==========
    
    def calculate_order_flow_imbalance(self, bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> np.ndarray:
        """订单流失衡"""
        total_size = bid_sizes + ask_sizes
        return (bid_sizes - ask_sizes) / np.where(total_size == 0, 1, total_size)
    
    def calculate_bid_ask_spread_ratio(self, bid_prices: np.ndarray, ask_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """买卖价差比率"""
        spreads = ask_prices - bid_prices
        return spreads / close_prices
    
    def calculate_market_pressure(self, bid_sizes: np.ndarray, ask_sizes: np.ndarray, period: int = 20) -> np.ndarray:
        """市场压力指标"""
        pressure = (bid_sizes - ask_sizes) / (bid_sizes + ask_sizes)
        return talib.SMA(pressure, timeperiod=period)
    
    # ========== 实时计算接口 ==========
    
    def calculate_realtime_indicators(self, symbol: str) -> Dict[str, float]:
        """计算实时技术指标"""
        if symbol not in self.buffers:
            return {}
        
        try:
            buffers = self.buffers[symbol]
            indicators = {}
            
            # 检查数据充足性
            min_data_points = max(self.params['sma_periods'] + self.params['ema_periods'] + [self.params['bb_period']])
            if len(buffers['close']) < min_data_points:
                return {}
            
            # 获取数据数组
            close = buffers['close'].get_array()
            high = buffers['high'].get_array()
            low = buffers['low'].get_array()
            volume = buffers['volume'].get_array()
            
            # 基础移动平均
            for period in self.params['sma_periods']:
                if len(close) >= period:
                    sma = self.calculate_sma(close, period)
                    indicators[f'sma_{period}'] = sma[-1] if not np.isnan(sma[-1]) else None
            
            for period in self.params['ema_periods']:
                if len(close) >= period:
                    ema = self.calculate_ema(close, period)
                    indicators[f'ema_{period}'] = ema[-1] if not np.isnan(ema[-1]) else None
            
            # 动量指标
            if len(close) >= self.params['rsi_period']:
                rsi = self.calculate_rsi(close, self.params['rsi_period'])
                indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else None
            
            # MACD
            if len(close) >= self.params['macd_slow']:
                macd, signal, histogram = self.calculate_macd(close, self.params['macd_fast'], 
                                                            self.params['macd_slow'], self.params['macd_signal'])
                indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else None
                indicators['macd_signal'] = signal[-1] if not np.isnan(signal[-1]) else None
                indicators['macd_histogram'] = histogram[-1] if not np.isnan(histogram[-1]) else None
            
            # 布林带
            if len(close) >= self.params['bb_period']:
                bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, self.params['bb_period'], self.params['bb_std'])
                indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else None
                indicators['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else None
                indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else None
                
                # 布林带位置
                if indicators['bb_upper'] and indicators['bb_lower']:
                    bb_position = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
                    indicators['bb_position'] = bb_position
            
            # VWAP
            if len(close) >= 20:
                vwap = self.calculate_vwap(high, low, close, volume)
                indicators['vwap'] = vwap[-1] if not np.isnan(vwap[-1]) else None
            
            # ATR
            if len(close) >= self.params['atr_period']:
                atr = self.calculate_atr(high, low, close, self.params['atr_period'])
                indicators['atr'] = atr[-1] if not np.isnan(atr[-1]) else None
            
            # 随机指标
            if len(close) >= self.params['stoch_k']:
                stoch_k, stoch_d = self.calculate_stochastic(high, low, close, self.params['stoch_k'], self.params['stoch_d'])
                indicators['stoch_k'] = stoch_k[-1] if not np.isnan(stoch_k[-1]) else None
                indicators['stoch_d'] = stoch_d[-1] if not np.isnan(stoch_d[-1]) else None
            
            # 费雪变换
            if len(close) >= self.params['fisher_period']:
                fisher, fisher_signal = self.calculate_fisher_transform(high, low, self.params['fisher_period'])
                indicators['fisher'] = fisher[-1] if not np.isnan(fisher[-1]) else None
                indicators['fisher_signal'] = fisher_signal[-1] if not np.isnan(fisher_signal[-1]) else None
            
            # Gamma
            if len(close) >= self.params['gamma_period']:
                gamma = self.calculate_gamma(close, self.params['gamma_period'])
                indicators['gamma'] = gamma[-1] if not np.isnan(gamma[-1]) else None
            
            # 订单薄指标（如果有数据）
            if 'bid_size' in buffers and 'ask_size' in buffers:
                bid_sizes = buffers['bid_size'].get_array()
                ask_sizes = buffers['ask_size'].get_array()
                
                if len(bid_sizes) > 0 and len(ask_sizes) > 0:
                    ofi = self.calculate_order_flow_imbalance(bid_sizes, ask_sizes)
                    indicators['order_flow_imbalance'] = ofi[-1] if not np.isnan(ofi[-1]) else None
                    
                    market_pressure = self.calculate_market_pressure(bid_sizes, ask_sizes)
                    indicators['market_pressure'] = market_pressure[-1] if not np.isnan(market_pressure[-1]) else None
            
            return {k: v for k, v in indicators.items() if v is not None}
            
        except Exception as e:
            self.logger.error(f"实时指标计算失败 {symbol}: {e}")
            return {}
    
    def get_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标（批量模式）"""
        try:
            result_df = df.copy()
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values if 'volume' in df.columns else None
            
            # 移动平均
            for period in self.params['sma_periods']:
                result_df[f'sma_{period}'] = self.calculate_sma(close, period)
            
            for period in self.params['ema_periods']:
                result_df[f'ema_{period}'] = self.calculate_ema(close, period)
            
            # 动量指标
            result_df['rsi'] = self.calculate_rsi(close, self.params['rsi_period'])
            
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            result_df['macd'] = macd
            result_df['macd_signal'] = macd_signal
            result_df['macd_histogram'] = macd_hist
            
            # 波动率指标
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            result_df['bb_upper'] = bb_upper
            result_df['bb_middle'] = bb_middle
            result_df['bb_lower'] = bb_lower
            
            result_df['atr'] = self.calculate_atr(high, low, close)
            
            # 趋势指标
            result_df['adx'] = self.calculate_adx(high, low, close)
            
            # 成交量指标
            if volume is not None:
                result_df['vwap'] = self.calculate_vwap(high, low, close, volume)
                result_df['obv'] = self.calculate_obv(close, volume)
                result_df['mfi'] = self.calculate_mfi(high, low, close, volume)
            
            # 高级指标
            fisher, fisher_signal = self.calculate_fisher_transform(high, low)
            result_df['fisher'] = fisher
            result_df['fisher_signal'] = fisher_signal
            
            result_df['gamma'] = self.calculate_gamma(close)
            result_df['connors_rsi'] = self.calculate_connors_rsi(close)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"批量指标计算失败: {e}")
            return df

