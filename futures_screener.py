# futures_screener.py - 期货筛选器
# =============================================================================
# 核心职责：
# 1. 流动性分析
# 2. 波动率筛选
# 3. 相关性计算
# 4. 合约优选
# 5. 动态调整
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from database_manager import DatabaseManager
from technical_indicators import TechnicalIndicators


@dataclass
class ContractInfo:
    """合约信息"""
    symbol: str
    exchange: str
    product: str
    size: float
    pricetick: float
    margin_rate: float
    commission: float
    
    # 统计信息
    avg_volume: float = 0.0
    avg_spread: float = 0.0
    avg_volatility: float = 0.0
    liquidity_score: float = 0.0
    
    # 相关性信息
    correlations: Dict[str, float] = field(default_factory=dict)
    
    # 评分
    total_score: float = 0.0


@dataclass
class ScreeningCriteria:
    """筛选条件"""
    min_volume: float = 10000
    min_liquidity_score: float = 0.6
    max_spread_ratio: float = 0.002
    min_volatility: float = 0.005
    max_volatility: float = 0.05
    max_correlation: float = 0.8
    lookback_days: int = 30


class FuturesScreener:
    """
    期货筛选器
    
    核心功能：
    1. 基于流动性和波动率筛选
    2. 计算品种间相关性
    3. 动态评分和排序
    4. 定期更新筛选结果
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("FUTURES_SCREENER")
        self.db_manager = DatabaseManager(self.config)
        self.tech_indicators = TechnicalIndicators(config_path)
        
        # 合约信息
        self.contracts: Dict[str, ContractInfo] = {}
        self.selected_contracts: List[str] = []
        
        # 筛选条件
        self.criteria = self._load_criteria()
        
        # 更新控制
        self.last_update = None
        self.update_interval = self.config.get("screener.update_interval", 3600)  # 秒
        
        self.logger.info("期货筛选器初始化完成")
    
    def _load_criteria(self) -> ScreeningCriteria:
        """加载筛选条件"""
        return ScreeningCriteria(
            min_volume=self.config.get("screener.min_volume", 10000),
            min_liquidity_score=self.config.get("screener.min_liquidity_score", 0.6),
            max_spread_ratio=self.config.get("screener.max_spread_ratio", 0.002),
            min_volatility=self.config.get("screener.min_volatility", 0.005),
            max_volatility=self.config.get("screener.max_volatility", 0.05),
            max_correlation=self.config.get("screener.max_correlation", 0.8),
            lookback_days=self.config.get("screener.lookback_days", 30)
        )
    
    async def initialize(self, contracts: List[Dict[str, Any]]):
        """初始化合约信息"""
        for contract_data in contracts:
            contract = ContractInfo(
                symbol=contract_data['symbol'],
                exchange=contract_data['exchange'],
                product=contract_data['product'],
                size=contract_data['size'],
                pricetick=contract_data['pricetick'],
                margin_rate=contract_data.get('margin_rate', 0.1),
                commission=contract_data.get('commission', 0.0001)
            )
            self.contracts[contract.symbol] = contract
        
        self.logger.info(f"加载了 {len(self.contracts)} 个合约")
    
    async def screen_contracts(self, force_update: bool = False) -> List[str]:
        """筛选合约"""
        # 检查是否需要更新
        if not force_update and self._is_cache_valid():
            return self.selected_contracts
        
        self.logger.info("开始筛选期货合约...")
        
        # 1. 计算流动性指标
        await self._calculate_liquidity_metrics()
        
        # 2. 计算波动率
        await self._calculate_volatility()
        
        # 3. 计算相关性
        await self._calculate_correlations()
        
        # 4. 综合评分
        self._calculate_scores()
        
        # 5. 筛选和排序
        self.selected_contracts = self._select_contracts()
        
        self.last_update = datetime.now()
        
        self.logger.info(f"筛选完成，选中 {len(self.selected_contracts)} 个合约")
        return self.selected_contracts
    
    async def _calculate_liquidity_metrics(self):
        """计算流动性指标"""
        end_time = TimeUtils.now_timestamp()
        start_time = end_time - self.criteria.lookback_days * 24 * 3600 * 1000
        
        for symbol, contract in self.contracts.items():
            try:
                # 获取tick数据
                tick_data = await self.db_manager.query_tick_data(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not tick_data.empty:
                    # 平均成交量
                    contract.avg_volume = tick_data['volume'].mean()
                    
                    # 平均价差
                    spreads = tick_data['ask_price_1'] - tick_data['bid_price_1']
                    mid_prices = (tick_data['ask_price_1'] + tick_data['bid_price_1']) / 2
                    spread_ratios = spreads / mid_prices
                    contract.avg_spread = spread_ratios.mean()
                    
                    # 流动性评分
                    volume_score = min(contract.avg_volume / 50000, 1.0)
                    spread_score = max(0, 1 - contract.avg_spread / 0.005)
                    contract.liquidity_score = 0.7 * volume_score + 0.3 * spread_score
                
            except Exception as e:
                self.logger.error(f"计算 {symbol} 流动性指标失败: {e}")
    
    async def _calculate_volatility(self):
        """计算波动率"""
        end_time = TimeUtils.now_timestamp()
        start_time = end_time - self.criteria.lookback_days * 24 * 3600 * 1000
        
        for symbol, contract in self.contracts.items():
            try:
                # 获取日线数据
                bar_data = await self.db_manager.query_bar_data(
                    symbol=symbol,
                    interval='1d',
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not bar_data.empty:
                    # 计算收益率
                    returns = bar_data['close'].pct_change().dropna()
                    
                    # 历史波动率（年化）
                    contract.avg_volatility = returns.std() * np.sqrt(252)
                
            except Exception as e:
                self.logger.error(f"计算 {symbol} 波动率失败: {e}")
    
    async def _calculate_correlations(self):
        """计算品种间相关性"""
        end_time = TimeUtils.now_timestamp()
        start_time = end_time - self.criteria.lookback_days * 24 * 3600 * 1000
        
        # 获取所有品种的收益率
        returns_dict = {}
        
        for symbol in self.contracts:
            try:
                bar_data = await self.db_manager.query_bar_data(
                    symbol=symbol,
                    interval='1d',
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not bar_data.empty:
                    returns = bar_data['close'].pct_change().dropna()
                    returns_dict[symbol] = returns
                    
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {e}")
        
        # 计算相关性矩阵
        if len(returns_dict) > 1:
            returns_df = pd.DataFrame(returns_dict)
            corr_matrix = returns_df.corr()
            
            # 存储相关性
            for symbol1 in corr_matrix.index:
                if symbol1 in self.contracts:
                    for symbol2 in corr_matrix.columns:
                        if symbol1 != symbol2:
                            correlation = corr_matrix.loc[symbol1, symbol2]
                            self.contracts[symbol1].correlations[symbol2] = correlation
    
    def _calculate_scores(self):
        """计算综合评分"""
        for contract in self.contracts.values():
            # 流动性得分 (40%)
            liquidity_score = contract.liquidity_score * 0.4
            
            # 波动率得分 (30%)
            if self.criteria.min_volatility <= contract.avg_volatility <= self.criteria.max_volatility:
                volatility_score = 0.3
            else:
                volatility_score = 0.0
            
            # 相关性得分 (20%)
            max_corr = max(contract.correlations.values()) if contract.correlations else 0
            corr_score = max(0, (1 - max_corr)) * 0.2
            
            # 成本得分 (10%)
            cost_score = max(0, 1 - contract.commission * 10000) * 0.1
            
            # 总分
            contract.total_score = liquidity_score + volatility_score + corr_score + cost_score
    
    def _select_contracts(self) -> List[str]:
        """选择合约"""
        # 筛选符合条件的合约
        candidates = []
        
        for symbol, contract in self.contracts.items():
            # 基本筛选条件
            if (contract.avg_volume >= self.criteria.min_volume and
                contract.liquidity_score >= self.criteria.min_liquidity_score and
                contract.avg_spread <= self.criteria.max_spread_ratio and
                self.criteria.min_volatility <= contract.avg_volatility <= self.criteria.max_volatility):
                
                candidates.append((symbol, contract))
        
        # 按评分排序
        candidates.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # 选择相关性低的品种
        selected = []
        
        for symbol, contract in candidates:
            # 检查与已选品种的相关性
            can_select = True
            
            for selected_symbol in selected:
                correlation = contract.correlations.get(selected_symbol, 0)
                if abs(correlation) > self.criteria.max_correlation:
                    can_select = False
                    break
            
            if can_select:
                selected.append(symbol)
            
            # 限制最大数量
            max_contracts = self.config.get("screener.max_contracts", 10)
            if len(selected) >= max_contracts:
                break
        
        return selected
    
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if not self.last_update:
            return False
        
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < self.update_interval
    
    def get_contract_info(self, symbol: str) -> Optional[ContractInfo]:
        """获取合约信息"""
        return self.contracts.get(symbol)
    
    def get_screening_report(self) -> Dict[str, Any]:
        """获取筛选报告"""
        report = {
            'update_time': self.last_update.isoformat() if self.last_update else None,
            'total_contracts': len(self.contracts),
            'selected_contracts': len(self.selected_contracts),
            'criteria': {
                'min_volume': self.criteria.min_volume,
                'min_liquidity_score': self.criteria.min_liquidity_score,
                'volatility_range': [self.criteria.min_volatility, self.criteria.max_volatility],
                'max_correlation': self.criteria.max_correlation
            },
            'selected': []
        }
        
        # 添加选中合约详情
        for symbol in self.selected_contracts:
            contract = self.contracts.get(symbol)
            if contract:
                report['selected'].append({
                    'symbol': symbol,
                    'score': contract.total_score,
                    'liquidity_score': contract.liquidity_score,
                    'avg_volume': contract.avg_volume,
                    'avg_volatility': contract.avg_volatility,
                    'avg_spread': contract.avg_spread
                })
        
        return report
    
    async def update_realtime_metrics(self, symbol: str, tick_data: Dict[str, Any]):
        """更新实时指标"""
        if symbol not in self.contracts:
            return
        
        contract = self.contracts[symbol]
        
        # 更新实时流动性指标
        # 这里可以实现更复杂的实时更新逻辑
        pass
