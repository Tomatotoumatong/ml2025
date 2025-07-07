# data_cleaner.py - 数据清洗
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from logger import TradingLogger
from utils import ConfigManager, DataUtils


class OutlierDetector:
    """异常值检测器"""
    
    @staticmethod
    def detect_iqr(data: pd.Series, factor: float = 1.5) -> np.ndarray:
        """IQR方法检测异常值"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def detect_zscore(data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Z-Score方法检测异常值"""
        z_scores = np.abs(stats.zscore(data.dropna()))
        outlier_mask = np.zeros(len(data), dtype=bool)
        outlier_mask[data.notna()] = z_scores > threshold
        return outlier_mask
    
    @staticmethod
    def detect_modified_zscore(data: pd.Series, threshold: float = 3.5) -> np.ndarray:
        """修正Z-Score方法检测异常值"""
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    @staticmethod
    def detect_price_outliers(
        df: pd.DataFrame,
        price_cols: List[str] = ['open', 'high', 'low', 'close'],
        volume_col: str = 'volume',
        method: str = 'iqr'
    ) -> Dict[str, np.ndarray]:
        """检测价格数据异常值"""
        outliers = {}
        
        for col in price_cols:
            if col in df.columns:
                if method == 'iqr':
                    outliers[col] = OutlierDetector.detect_iqr(df[col])
                elif method == 'zscore':
                    outliers[col] = OutlierDetector.detect_zscore(df[col])
                elif method == 'modified_zscore':
                    outliers[col] = OutlierDetector.detect_modified_zscore(df[col])
        
        # 成交量异常值检测
        if volume_col in df.columns:
            outliers[volume_col] = OutlierDetector.detect_modified_zscore(df[volume_col])
        
        # 价格一致性检查
        if all(col in df.columns for col in ['high', 'low', 'close']):
            price_inconsistent = (df['high'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])
            outliers['price_inconsistent'] = price_inconsistent.values
        
        return outliers


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("DATA_CLEANER")
        
        # 清洗配置
        self.outlier_method = self.config.get("cleaner.outlier_method", "modified_zscore")
        self.outlier_threshold = self.config.get("cleaner.outlier_threshold", 3.5)
        self.imputation_method = self.config.get("cleaner.imputation_method", "linear")
        self.scaling_method = self.config.get("cleaner.scaling_method", "robust")
        
        # 异常值处理策略
        self.outlier_strategy = self.config.get("cleaner.outlier_strategy", "cap")  # cap, remove, interpolate
        
        # 初始化缩放器
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        self.fitted_scalers: Dict[str, Any] = {}
    
    def clean_market_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        detect_outliers: bool = True,
        handle_missing: bool = True,
        normalize_data: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        清洗市场数据
        
        Args:
            df: 原始数据
            symbol: 交易对
            detect_outliers: 是否检测异常值
            handle_missing: 是否处理缺失值
            normalize_data: 是否标准化数据
        
        Returns:
            清洗后的数据和质量报告
        """
        original_shape = df.shape
        quality_report = {
            'symbol': symbol,
            'original_rows': original_shape[0],
            'original_cols': original_shape[1],
            'cleaning_steps': [],
            'outliers_detected': {},
            'missing_values_filled': {},
            'data_quality_score': 0.0
        }
        
        df_cleaned = df.copy()
        
        try:
            # 1. 基础数据验证
            df_cleaned = self._validate_basic_data(df_cleaned, quality_report)
            
            # 2. 异常值检测和处理
            if detect_outliers:
                df_cleaned = self._handle_outliers(df_cleaned, quality_report)
            
            # 3. 缺失值处理
            if handle_missing:
                df_cleaned = self._handle_missing_values(df_cleaned, quality_report)
            
            # 4. 数据标准化
            if normalize_data:
                df_cleaned = self._normalize_data(df_cleaned, symbol, quality_report)
            
            # 5. 最终质量评估
            quality_report['final_rows'] = len(df_cleaned)
            quality_report['data_quality_score'] = self._calculate_quality_score(df_cleaned, quality_report)
            
            self.logger.info(f"{symbol} 数据清洗完成 - 质量评分: {quality_report['data_quality_score']:.2f}")
            
            return df_cleaned, quality_report
            
        except Exception as e:
            self.logger.error(f"数据清洗失败 {symbol}: {e}")
            return df, quality_report
    
    def _validate_basic_data(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> pd.DataFrame:
        """基础数据验证"""
        initial_rows = len(df)
        
        # 移除完全重复的行
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        if duplicates_removed > 0:
            quality_report['cleaning_steps'].append(f"移除重复行: {duplicates_removed}")
        
        # 时间序列排序
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # 价格数据基础验证
        price_cols = ['open', 'high', 'low', 'close']
        valid_price_cols = [col for col in price_cols if col in df.columns]
        
        for col in valid_price_cols:
            # 移除非正数价格
            invalid_prices = (df[col] <= 0) | (~np.isfinite(df[col]))
            if invalid_prices.any():
                df = df[~invalid_prices]
                quality_report['cleaning_steps'].append(f"移除无效{col}价格: {invalid_prices.sum()}")
        
        # 成交量验证
        if 'volume' in df.columns:
            invalid_volume = (df['volume'] < 0) | (~np.isfinite(df['volume']))
            if invalid_volume.any():
                df = df[~invalid_volume]
                quality_report['cleaning_steps'].append(f"移除无效成交量: {invalid_volume.sum()}")
        
        return df.reset_index(drop=True)
    
    def _handle_outliers(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> pd.DataFrame:
        """异常值处理"""
        price_cols = ['open', 'high', 'low', 'close']
        volume_col = 'volume'
        
        # 检测异常值
        outliers = OutlierDetector.detect_price_outliers(
            df, price_cols, volume_col, self.outlier_method
        )
        
        quality_report['outliers_detected'] = {
            col: int(mask.sum()) for col, mask in outliers.items()
        }
        
        # 处理异常值
        for col, outlier_mask in outliers.items():
            if outlier_mask.any() and col in df.columns:
                outlier_count = outlier_mask.sum()
                
                if self.outlier_strategy == 'remove':
                    df = df[~outlier_mask]
                    quality_report['cleaning_steps'].append(f"移除{col}异常值: {outlier_count}")
                
                elif self.outlier_strategy == 'cap':
                    if col != 'price_inconsistent':
                        # 用分位数截断
                        q1, q99 = df[col].quantile([0.01, 0.99])
                        df.loc[outlier_mask, col] = np.clip(df.loc[outlier_mask, col], q1, q99)
                        quality_report['cleaning_steps'].append(f"截断{col}异常值: {outlier_count}")
                
                elif self.outlier_strategy == 'interpolate':
                    df.loc[outlier_mask, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
                    quality_report['cleaning_steps'].append(f"插值{col}异常值: {outlier_count}")
        
        return df.reset_index(drop=True)
    
    def _handle_missing_values(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> pd.DataFrame:
        """缺失值处理"""
        missing_before = df.isnull().sum()
        columns_with_missing = missing_before[missing_before > 0].index.tolist()
        
        if not columns_with_missing:
            return df
        
        for col in columns_with_missing:
            missing_count = missing_before[col]
            missing_ratio = missing_count / len(df)
            
            # 如果缺失比例过高，考虑删除列
            if missing_ratio > 0.5:
                df = df.drop(columns=[col])
                quality_report['cleaning_steps'].append(f"删除缺失率过高的列{col}: {missing_ratio:.2%}")
                continue
            
            # 根据列类型选择填充方法
            if col in ['open', 'high', 'low', 'close']:
                # 价格数据用线性插值
                df[col] = df[col].interpolate(method='linear')
                
            elif col == 'volume':
                # 成交量用前向填充
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
            else:
                # 其他数据用中位数填充
                df[col] = df[col].fillna(df[col].median())
            
            filled_count = missing_count - df[col].isnull().sum()
            quality_report['missing_values_filled'][col] = int(filled_count)
        
        # 删除仍有缺失值的行
        remaining_missing = df.isnull().any(axis=1).sum()
        if remaining_missing > 0:
            df = df.dropna()
            quality_report['cleaning_steps'].append(f"删除剩余缺失值行: {remaining_missing}")
        
        return df.reset_index(drop=True)
    
    def _normalize_data(self, df: pd.DataFrame, symbol: str, quality_report: Dict[str, Any]) -> pd.DataFrame:
        """数据标准化"""
        normalize_cols = []
        
        # 确定需要标准化的列
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                normalize_cols.append(col)
        
        if 'volume' in df.columns:
            normalize_cols.append('volume')
        
        if not normalize_cols:
            return df
        
        # 选择缩放器
        scaler = self.scalers[self.scaling_method].copy()
        
        # 标准化数据
        try:
            df_scaled = df.copy()
            scaled_values = scaler.fit_transform(df[normalize_cols])
            df_scaled[normalize_cols] = scaled_values
            
            # 保存缩放器用于实时数据处理
            self.fitted_scalers[symbol] = scaler
            
            quality_report['cleaning_steps'].append(f"标准化列: {normalize_cols}")
            
            return df_scaled
            
        except Exception as e:
            self.logger.error(f"数据标准化失败: {e}")
            return df
    
    def _calculate_quality_score(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> float:
        """计算数据质量评分"""
        score = 100.0
        
        # 根据数据完整性扣分
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 50
        
        # 根据异常值比例扣分
        total_outliers = sum(quality_report.get('outliers_detected', {}).values())
        outlier_ratio = total_outliers / len(df) if len(df) > 0 else 0
        score -= outlier_ratio * 30
        
        # 根据数据量扣分
        if len(df) < 100:
            score -= 20
        elif len(df) < 1000:
            score -= 10
        
        return max(0.0, min(100.0, score))
    
    def transform_realtime_data(self, data: Dict[str, float], symbol: str) -> Dict[str, float]:
        """转换实时数据（使用已训练的缩放器）"""
        if symbol not in self.fitted_scalers:
            return data
        
        try:
            scaler = self.fitted_scalers[symbol]
            
            # 提取需要标准化的特征
            price_cols = ['open', 'high', 'low', 'close']
            feature_cols = [col for col in price_cols if col in data]
            
            if 'volume' in data:
                feature_cols.append('volume')
            
            if not feature_cols:
                return data
            
            # 构建特征数组
            features = np.array([[data[col] for col in feature_cols]])
            
            # 标准化
            scaled_features = scaler.transform(features)[0]
            
            # 更新数据
            scaled_data = data.copy()
            for i, col in enumerate(feature_cols):
                scaled_data[col] = float(scaled_features[i])
            
            return scaled_data
            
        except Exception as e:
            self.logger.error(f"实时数据转换失败 {symbol}: {e}")
            return data
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'time_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'price_summary': df[['open', 'high', 'low', 'close']].describe().to_dict() if all(col in df.columns for col in ['open', 'high', 'low', 'close']) else {}
        }


# 集成测试函数
async def test_data_layer():
    """测试数据层功能"""
    logger = TradingLogger().get_logger("DATA_LAYER_TEST")
    logger.info("🧪 开始数据层测试...")
    
    # 测试数据采集器
    collector = MarketDataCollector()
    
    # 添加数据回调
    async def data_callback(data_type: str, data: Dict[str, Any]):
        logger.info(f"接收到{data_type}数据: {data.get('symbol', 'UNKNOWN')}")
    
    collector.add_data_callback(data_callback)
    
    # 测试数据清洗器
    cleaner = DataCleaner()
    
    # 生成测试数据
    test_data = pd.DataFrame({
        'timestamp': range(1000),
        'open': np.random.normal(50000, 1000, 1000),
        'high': np.random.normal(51000, 1000, 1000),
        'low': np.random.normal(49000, 1000, 1000),
        'close': np.random.normal(50000, 1000, 1000),
        'volume': np.random.exponential(1000, 1000)
    })
    
    # 添加一些异常值和缺失值
    test_data.loc[10:15, 'close'] = np.nan
    test_data.loc[50, 'high'] = 100000  # 异常值
    test_data.loc[100, 'volume'] = -100  # 无效值
    
    logger.info("清洗测试数据...")
    cleaned_data, quality_report = cleaner.clean_market_data(test_data, "BTCUSDT")
    
    logger.info(f"清洗结果: {quality_report['data_quality_score']:.2f}分")
    logger.info(f"处理步骤: {quality_report['cleaning_steps']}")
    
    logger.info("✅ 数据层测试完成")

if __name__ == "__main__":
    asyncio.run(test_data_layer())