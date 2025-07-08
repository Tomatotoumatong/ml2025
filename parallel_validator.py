# parallel_validator.py - 并行验证器
# =============================================================================
# 核心职责：
# 1. 演习场模式实现
# 2. A/B测试框架
# 3. 策略对比分析
# 4. 实时性能评估
# 5. 结果可视化
# =============================================================================

import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from paper_trading_validator import PaperTradingValidator, PerformanceMetrics
from vnpy_integration import TradingSignal
from pathlib import Path

@dataclass
class StrategyInstance:
    """策略实例"""
    name: str
    version: str
    validator: PaperTradingValidator
    config: Dict[str, Any]
    is_production: bool = False
    is_active: bool = True
    
    # 统计信息
    signals_generated: int = 0
    orders_executed: int = 0
    last_signal_time: Optional[int] = None
    
    # 性能指标
    metrics: Optional[PerformanceMetrics] = None


@dataclass
class ComparisonResult:
    """对比结果"""
    timestamp: int
    metrics: Dict[str, PerformanceMetrics]
    winner: Optional[str] = None
    confidence: float = 0.0
    analysis: Dict[str, Any] = field(default_factory=dict)


class ParallelValidator:
    """
    并行验证器
    
    核心功能：
    1. 多策略并行运行
    2. 演习场模式管理
    3. A/B测试执行
    4. 实时对比分析
    5. 自动策略切换
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("PARALLEL_VALIDATOR")
        
        # 策略实例
        self.strategies: Dict[str, StrategyInstance] = {}
        self.production_strategy: Optional[str] = None
        
        # 演习场配置
        self.max_parallel_strategies = self.config.get(
            "parallel_validator.max_strategies", 5
        )
        self.comparison_interval = self.config.get(
            "parallel_validator.comparison_interval", 3600  # 1小时
        )
        self.promotion_threshold = self.config.get(
            "parallel_validator.promotion_threshold", 0.95  # 95%置信度
        )
        self.min_test_duration = self.config.get(
            "parallel_validator.min_test_duration", 86400  # 24小时
        )
        
        # 对比结果
        self.comparison_history: List[ComparisonResult] = []
        self.last_comparison_time = None
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_strategies)
        
        # 回调函数
        self.signal_router: Optional[Callable] = None
        self.promotion_callback: Optional[Callable] = None
        
        # 运行控制
        self.is_running = False
        self.comparison_task = None
        
        self.logger.info("并行验证器初始化完成")
    
    async def add_strategy(
        self, 
        name: str, 
        version: str,
        config: Dict[str, Any],
        is_production: bool = False
    ) -> bool:
        """添加策略"""
        try:
            if len(self.strategies) >= self.max_parallel_strategies:
                self.logger.warning(f"已达到最大策略数量限制: {self.max_parallel_strategies}")
                return False
            
            # 创建验证器
            validator = PaperTradingValidator(self.config.config_path)
            await validator.start()
            
            # 创建策略实例
            strategy = StrategyInstance(
                name=name,
                version=version,
                validator=validator,
                config=config,
                is_production=is_production
            )
            
            self.strategies[name] = strategy
            
            if is_production:
                self.production_strategy = name
            
            self.logger.info(f"添加策略: {name} v{version} (生产: {is_production})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"添加策略失败: {e}")
            return False
    
    async def remove_strategy(self, name: str) -> bool:
        """移除策略"""
        if name not in self.strategies:
            return False
        
        if name == self.production_strategy:
            self.logger.warning("不能移除生产策略")
            return False
        
        strategy = self.strategies[name]
        strategy.is_active = False
        
        del self.strategies[name]
        
        self.logger.info(f"移除策略: {name}")
        return True
    
    async def start(self):
        """启动验证器"""
        self.is_running = True
        
        # 启动对比任务
        self.comparison_task = asyncio.create_task(self._comparison_loop())
        
        self.logger.info("并行验证器已启动")
    
    async def stop(self):
        """停止验证器"""
        self.is_running = False
        
        if self.comparison_task:
            self.comparison_task.cancel()
            try:
                await self.comparison_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("并行验证器已停止")
    
    async def route_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """路由信号到各策略"""
        if not self.strategies:
            return None
        
        # 并行生成信号
        tasks = []
        
        for name, strategy in self.strategies.items():
            if strategy.is_active:
                task = asyncio.create_task(
                    self._generate_signal_for_strategy(name, strategy, market_data)
                )
                tasks.append((name, task))
        
        # 等待所有策略完成
        signals = {}
        
        for name, task in tasks:
            try:
                signal = await task
                if signal:
                    signals[name] = signal
            except Exception as e:
                self.logger.error(f"策略 {name} 生成信号失败: {e}")
        
        # 返回生产策略的信号
        if self.production_strategy and self.production_strategy in signals:
            return signals[self.production_strategy]
        
        return None
    
    async def _generate_signal_for_strategy(
        self,
        name: str,
        strategy: StrategyInstance,
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """为特定策略生成信号"""
        try:
            # 调用信号生成回调
            if self.signal_router:
                signal = await self.signal_router(
                    strategy_name=name,
                    config=strategy.config,
                    market_data=market_data
                )
                
                if signal:
                    strategy.signals_generated += 1
                    strategy.last_signal_time = TimeUtils.now_timestamp()
                    
                    # 在验证器中执行
                    result = await strategy.validator.execute_signal(signal)
                    
                    if result.status in ['filled', 'partial_filled']:
                        strategy.orders_executed += 1
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"策略 {name} 信号生成异常: {e}")
            return None
    
    async def _comparison_loop(self):
        """对比循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.comparison_interval)
                
                # 执行对比
                comparison = await self._compare_strategies()
                
                if comparison:
                    self.comparison_history.append(comparison)
                    
                    # 检查是否需要晋升策略
                    await self._check_promotion(comparison)
                
                self.last_comparison_time = datetime.now()
                
            except Exception as e:
                self.logger.error(f"对比循环异常: {e}")
    
    async def _compare_strategies(self) -> Optional[ComparisonResult]:
        """对比策略"""
        if len(self.strategies) < 2:
            return None
        
        try:
            # 收集各策略指标
            metrics_dict = {}
            
            for name, strategy in self.strategies.items():
                if strategy.is_active:
                    metrics = strategy.validator.calculate_performance()
                    strategy.metrics = metrics
                    metrics_dict[name] = metrics
            
            # 分析对比结果
            analysis = self._analyze_comparison(metrics_dict)
            
            # 确定获胜者
            winner, confidence = self._determine_winner(metrics_dict, analysis)
            
            comparison = ComparisonResult(
                timestamp=TimeUtils.now_timestamp(),
                metrics=metrics_dict,
                winner=winner,
                confidence=confidence,
                analysis=analysis
            )
            
            # 记录日志
            self.logger.info(
                f"策略对比完成 - 获胜者: {winner} "
                f"(置信度: {confidence:.2%})"
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"策略对比失败: {e}")
            return None
    
    def _analyze_comparison(
        self, 
        metrics_dict: Dict[str, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """分析对比结果"""
        analysis = {
            'best_return': None,
            'best_sharpe': None,
            'best_win_rate': None,
            'lowest_drawdown': None,
            'rankings': {}
        }
        
        # 按各指标排名
        if metrics_dict:
            # 总收益率排名
            returns = [(name, m.total_return) for name, m in metrics_dict.items()]
            returns.sort(key=lambda x: x[1], reverse=True)
            analysis['best_return'] = returns[0][0] if returns else None
            analysis['rankings']['return'] = [name for name, _ in returns]
            
            # 夏普比率排名
            sharpes = [(name, m.sharpe_ratio) for name, m in metrics_dict.items()]
            sharpes.sort(key=lambda x: x[1], reverse=True)
            analysis['best_sharpe'] = sharpes[0][0] if sharpes else None
            analysis['rankings']['sharpe'] = [name for name, _ in sharpes]
            
            # 胜率排名
            win_rates = [(name, m.win_rate) for name, m in metrics_dict.items()]
            win_rates.sort(key=lambda x: x[1], reverse=True)
            analysis['best_win_rate'] = win_rates[0][0] if win_rates else None
            analysis['rankings']['win_rate'] = [name for name, _ in win_rates]
            
            # 最大回撤排名（越小越好）
            drawdowns = [(name, m.max_drawdown) for name, m in metrics_dict.items()]
            drawdowns.sort(key=lambda x: x[1])
            analysis['lowest_drawdown'] = drawdowns[0][0] if drawdowns else None
            analysis['rankings']['drawdown'] = [name for name, _ in drawdowns]
        
        return analysis
    
    def _determine_winner(
        self,
        metrics_dict: Dict[str, PerformanceMetrics],
        analysis: Dict[str, Any]
    ) -> Tuple[Optional[str], float]:
        """确定获胜者"""
        if not metrics_dict:
            return None, 0.0
        
        # 计算综合得分
        scores = {}
        
        for name, metrics in metrics_dict.items():
            # 各指标权重
            weights = {
                'return': 0.3,
                'sharpe': 0.3,
                'win_rate': 0.2,
                'drawdown': 0.2
            }
            
            score = 0.0
            
            # 收益率得分
            if metrics.total_return > 0:
                score += weights['return'] * min(metrics.total_return / 0.5, 1.0)
            
            # 夏普比率得分
            if metrics.sharpe_ratio > 0:
                score += weights['sharpe'] * min(metrics.sharpe_ratio / 2.0, 1.0)
            
            # 胜率得分
            score += weights['win_rate'] * metrics.win_rate
            
            # 回撤得分（反向）
            score += weights['drawdown'] * (1 - min(metrics.max_drawdown / 0.2, 1.0))
            
            scores[name] = score
        
        # 找出最高分
        if scores:
            winner = max(scores, key=scores.get)
            winner_score = scores[winner]
            
            # 计算置信度
            if len(scores) > 1:
                other_scores = [s for n, s in scores.items() if n != winner]
                avg_other_score = np.mean(other_scores)
                
                # 置信度基于分数差异
                if avg_other_score > 0:
                    confidence = min((winner_score - avg_other_score) / avg_other_score, 1.0)
                else:
                    confidence = 1.0
            else:
                confidence = 0.5
            
            return winner, confidence
        
        return None, 0.0
    
    async def _check_promotion(self, comparison: ComparisonResult):
        """检查是否需要晋升策略"""
        if not comparison.winner:
            return
        
        # 检查获胜者是否已是生产策略
        if comparison.winner == self.production_strategy:
            return
        
        # 检查置信度
        if comparison.confidence < self.promotion_threshold:
            return
        
        # 检查测试时长
        winner_strategy = self.strategies.get(comparison.winner)
        if not winner_strategy:
            return
        
        # 计算测试时长
        test_duration = TimeUtils.now_timestamp() - (winner_strategy.last_signal_time or 0)
        
        if test_duration < self.min_test_duration * 1000:  # 转换为毫秒
            self.logger.info(
                f"策略 {comparison.winner} 测试时长不足: "
                f"{test_duration / 1000 / 3600:.1f} 小时"
            )
            return
        
        # 晋升策略
        await self._promote_strategy(comparison.winner)
    
    async def _promote_strategy(self, strategy_name: str):
        """晋升策略为生产策略"""
        try:
            if strategy_name not in self.strategies:
                return
            
            # 降级当前生产策略
            if self.production_strategy and self.production_strategy in self.strategies:
                self.strategies[self.production_strategy].is_production = False
            
            # 晋升新策略
            self.strategies[strategy_name].is_production = True
            old_production = self.production_strategy
            self.production_strategy = strategy_name
            
            self.logger.info(
                f"策略晋升: {old_production} -> {strategy_name}"
            )
            
            # 触发回调
            if self.promotion_callback:
                await self.promotion_callback(
                    old_strategy=old_production,
                    new_strategy=strategy_name
                )
                
        except Exception as e:
            self.logger.error(f"策略晋升失败: {e}")
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """获取对比报告"""
        report = {
            'active_strategies': len(self.strategies),
            'production_strategy': self.production_strategy,
            'last_comparison': self.last_comparison_time.isoformat() if self.last_comparison_time else None,
            'strategies': {}
        }
        
        # 添加各策略详情
        for name, strategy in self.strategies.items():
            strategy_info = {
                'version': strategy.version,
                'is_production': strategy.is_production,
                'is_active': strategy.is_active,
                'signals_generated': strategy.signals_generated,
                'orders_executed': strategy.orders_executed,
                'last_signal_time': TimeUtils.timestamp_to_str(strategy.last_signal_time) if strategy.last_signal_time else None
            }
            
            # 添加性能指标
            if strategy.metrics:
                strategy_info['performance'] = strategy.metrics.to_dict()
            
            report['strategies'][name] = strategy_info
        
        # 添加最新对比结果
        if self.comparison_history:
            latest = self.comparison_history[-1]
            report['latest_comparison'] = {
                'timestamp': TimeUtils.timestamp_to_str(latest.timestamp),
                'winner': latest.winner,
                'confidence': f"{latest.confidence:.2%}",
                'analysis': latest.analysis
            }
        
        return report
    
    def export_comparison_history(self, export_path: Path):
        """导出对比历史"""
        try:
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 导出对比历史
            history_data = []
            
            for comparison in self.comparison_history:
                record = {
                    'timestamp': comparison.timestamp,
                    'winner': comparison.winner,
                    'confidence': comparison.confidence
                }
                
                # 添加各策略指标
                for name, metrics in comparison.metrics.items():
                    record[f'{name}_return'] = metrics.total_return
                    record[f'{name}_sharpe'] = metrics.sharpe_ratio
                    record[f'{name}_drawdown'] = metrics.max_drawdown
                    record[f'{name}_win_rate'] = metrics.win_rate
                
                history_data.append(record)
            
            if history_data:
                df = pd.DataFrame(history_data)
                df.to_csv(export_path / "comparison_history.csv", index=False)
            
            # 导出各策略报告
            for name, strategy in self.strategies.items():
                strategy_path = export_path / name
                strategy.validator.export_results(strategy_path)
            
            self.logger.info(f"对比历史已导出到: {export_path}")
            
        except Exception as e:
            self.logger.error(f"导出对比历史失败: {e}")
    
    def set_signal_router(self, router: Callable):
        """设置信号路由器"""
        self.signal_router = router
    
    def set_promotion_callback(self, callback: Callable):
        """设置晋升回调"""
        self.promotion_callback = callback
