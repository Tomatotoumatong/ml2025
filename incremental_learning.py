# incremental_learning.py - 增量学习

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import copy
import hashlib

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from prepare_rl_data import TrajectoryBuffer
from market_environment import MarketState, MarketEnvironmentClassifier
from meta_model_pipeline import ModelType


@dataclass
class ModelSnapshot:
    """模型快照"""
    model_type: ModelType
    timestamp: int
    state_dict: Dict[str, torch.Tensor]
    performance_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]
    version: str
    
    def get_hash(self) -> str:
        """生成快照哈希"""
        content = f"{self.model_type.value}_{self.timestamp}_{self.version}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class ExperienceMemory:
    """经验记忆单元"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    market_state: MarketState
    importance: float = 1.0
    age: int = 0
    usage_count: int = 0
    
    def update_importance(self, td_error: float, novelty: float):
        """更新重要性分数"""
        # TD误差越大，重要性越高
        td_importance = abs(td_error) + 0.1
        
        # 新颖性越高，重要性越高
        novelty_importance = novelty + 0.1
        
        # 使用次数衰减
        usage_decay = 1.0 / (1 + self.usage_count * 0.1)
        
        # 时间衰减
        age_decay = 0.99 ** (self.age / 100)
        
        self.importance = td_importance * novelty_importance * usage_decay * age_decay


class IncrementalLearner:
    """
    增量学习管理器
    
    核心功能：
    1. 在线更新模型
    2. 管理模型版本
    3. 防止遗忘
    4. 知识蒸馏
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("INCREMENTAL_LEARNER")
        
        # 配置参数
        self.update_frequency = self.config.get("incremental.update_frequency", 100)
        self.batch_size = self.config.get("incremental.batch_size", 32)
        self.memory_size = self.config.get("incremental.memory_size", 10000)
        self.rehearsal_ratio = self.config.get("incremental.rehearsal_ratio", 0.3)
        self.elastic_weight = self.config.get("incremental.elastic_weight", 0.1)
        
        # 经验记忆
        self.episodic_memory = defaultdict(lambda: deque(maxlen=self.memory_size // 9))
        self.core_memories = deque(maxlen=self.memory_size // 10)  # 核心记忆
        
        # 模型快照管理
        self.model_snapshots: Dict[ModelType, deque] = {
            model_type: deque(maxlen=5)
            for model_type in ModelType
        }
        
        # Fisher信息矩阵（用于EWC）
        self.fisher_matrices: Dict[ModelType, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[ModelType, Dict[str, torch.Tensor]] = {}
        
        # 知识蒸馏
        self.teacher_models: Dict[ModelType, Any] = {}
        self.distillation_temperature = self.config.get("incremental.distillation_temp", 3.0)
        
        # 更新统计
        self.update_history = deque(maxlen=1000)
        self.forgetting_metrics = defaultdict(list)
        
        # 新颖性检测器
        self.novelty_threshold = self.config.get("incremental.novelty_threshold", 0.8)
        self.state_embeddings = deque(maxlen=5000)
    
    def should_update(self, model_type: ModelType) -> bool:
        """判断是否应该更新模型"""
        # 基于更新频率
        updates = [u for u in self.update_history if u['model_type'] == model_type]
        
        if not updates:
            return True
        
        last_update = updates[-1]['timestamp']
        time_since_update = TimeUtils.now_timestamp() - last_update
        
        # 时间触发
        if time_since_update > self.update_frequency * 60000:  # 转换为毫秒
            return True
        
        # 性能触发
        recent_performance = self._get_recent_performance(model_type)
        if recent_performance < 0.45:  # 性能下降
            return True
        
        # 新数据积累触发
        new_samples = self._count_new_samples(model_type, last_update)
        if new_samples > self.batch_size * 10:
            return True
        
        return False
    
    def update_model(self, 
                    model: Any,
                    model_type: ModelType,
                    new_experiences: List[Dict[str, Any]],
                    current_performance: Dict[str, float]) -> Any:
        """
        增量更新模型
        
        Args:
            model: 要更新的模型
            model_type: 模型类型
            new_experiences: 新经验
            current_performance: 当前性能
        
        Returns:
            更新后的模型
        """
        self.logger.info(f"开始增量更新 {model_type.value}")
        
        try:
            # 1. 保存当前模型快照
            self._save_model_snapshot(model, model_type, current_performance)
            
            # 2. 处理新经验
            processed_experiences = self._process_new_experiences(
                new_experiences, model_type
            )
            
            # 3. 构建训练批次（混合新旧经验）
            training_batch = self._construct_training_batch(
                processed_experiences, model_type
            )
            
            # 4. 计算Fisher信息（如果使用EWC）
            if self.elastic_weight > 0:
                self._update_fisher_information(model, training_batch)
            
            # 5. 执行增量更新
            updated_model = self._perform_incremental_update(
                model, model_type, training_batch
            )
            
            # 6. 验证更新效果
            if not self._validate_update(updated_model, model, model_type):
                self.logger.warning("更新验证失败，回滚到之前版本")
                return model
            
            # 7. 更新记忆和统计
            self._update_memories(processed_experiences, model_type)
            self._record_update(model_type, current_performance)
            
            return updated_model
            
        except Exception as e:
            self.logger.error(f"增量更新失败: {e}")
            return model
    
    def _process_new_experiences(self, 
                               experiences: List[Dict[str, Any]], 
                               model_type: ModelType) -> List[ExperienceMemory]:
        """处理新经验"""
        processed = []
        
        for exp in experiences:
            # 计算新颖性
            novelty = self._calculate_novelty(exp['state'])
            
            # 创建经验记忆
            memory = ExperienceMemory(
                state=exp['state'],
                action=exp['action'],
                reward=exp['reward'],
                next_state=exp['next_state'],
                market_state=exp.get('market_state', MarketState.SIDEWAYS),
                importance=novelty
            )
            
            # 高新颖性经验标记为重要
            if novelty > self.novelty_threshold:
                self.logger.debug(f"发现高新颖性经验: {novelty:.3f}")
                self.core_memories.append(memory)
            
            processed.append(memory)
        
        return processed
    
    def _construct_training_batch(self, 
                                new_experiences: List[ExperienceMemory],
                                model_type: ModelType) -> List[ExperienceMemory]:
        """构建训练批次"""
        batch = []
        
        # 添加新经验
        new_count = min(len(new_experiences), int(self.batch_size * (1 - self.rehearsal_ratio)))
        batch.extend(new_experiences[:new_count])
        
        # 添加旧经验（经验回放）
        old_count = self.batch_size - new_count
        
        # 优先从核心记忆中采样
        core_count = min(len(self.core_memories), old_count // 2)
        if core_count > 0:
            core_indices = np.random.choice(len(self.core_memories), core_count, replace=False)
            batch.extend([self.core_memories[i] for i in core_indices])
        
        # 从分类记忆中采样
        remaining = old_count - core_count
        for market_state in MarketState:
            if remaining <= 0:
                break
            
            state_memories = self.episodic_memory[market_state]
            if state_memories:
                sample_count = min(len(state_memories), remaining // 9)
                if sample_count > 0:
                    indices = np.random.choice(len(state_memories), sample_count, replace=False)
                    batch.extend([state_memories[i] for i in indices])
                    remaining -= sample_count
        
        # 更新使用计数和年龄
        for memory in batch:
            memory.usage_count += 1
            memory.age += 1
        
        return batch
    
    def _perform_incremental_update(self,
                                  model: Any,
                                  model_type: ModelType,
                                  training_batch: List[ExperienceMemory]) -> Any:
        """执行增量更新"""
        # 深拷贝模型
        updated_model = copy.deepcopy(model)
        
        # 根据模型类型执行不同的更新策略
        if model_type == ModelType.ML_ENSEMBLE:
            updated_model = self._update_ml_model(updated_model, training_batch)
        elif model_type in [ModelType.RL_PPO, ModelType.RL_A2C]:
            updated_model = self._update_rl_model(updated_model, training_batch)
        
        return updated_model
    
    def _update_ml_model(self, model: Any, batch: List[ExperienceMemory]) -> Any:
        """更新ML模型"""
        # ML模型的增量更新逻辑
        # 这里简化处理，实际应该调用ml_pipeline的增量训练方法
        self.logger.info("执行ML模型增量更新")
        
        # TODO: 实现具体的ML增量更新逻辑
        return model
    
    def _update_rl_model(self, model: Any, batch: List[ExperienceMemory]) -> Any:
        """更新RL模型"""
        if not hasattr(model, 'update'):
            return model
        
        # 准备训练数据
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        for memory in batch:
            trajectories['states'].append(memory.state)
            trajectories['actions'].append(memory.action)
            trajectories['rewards'].append(memory.reward)
            trajectories['next_states'].append(memory.next_state)
            trajectories['dones'].append(False)
        
        # 应用EWC约束
        if self.elastic_weight > 0 and model.model_type in self.fisher_matrices:
            self._apply_ewc_loss(model, trajectories)
        
        # 执行更新
        model.update(trajectories)
        
        return model
    
    def _update_fisher_information(self, model: Any, batch: List[ExperienceMemory]):
        """更新Fisher信息矩阵（用于EWC）"""
        if not hasattr(model, 'policy_net'):
            return
        
        fisher = {}
        model.policy_net.eval()
        
        # 计算每个参数的Fisher信息
        for name, param in model.policy_net.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        # 累积梯度平方
        for memory in batch:
            state_tensor = torch.FloatTensor(memory.state).unsqueeze(0)
            
            if hasattr(model, 'device'):
                state_tensor = state_tensor.to(model.device)
            
            # 前向传播
            action_probs, _ = model.policy_net(state_tensor)
            
            # 计算对数概率
            log_probs = torch.log(action_probs[0, memory.action])
            
            # 反向传播
            model.optimizer.zero_grad()
            log_probs.backward()
            
            # 累积梯度平方
            for name, param in model.policy_net.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # 平均
        for name in fisher:
            fisher[name] /= len(batch)
        
        self.fisher_matrices[model.model_type] = fisher
        
        # 保存当前最优参数
        self.optimal_params[model.model_type] = {
            name: param.data.clone()
            for name, param in model.policy_net.named_parameters()
        }
        
        model.policy_net.train()
    
    def _apply_ewc_loss(self, model: Any, trajectories: Dict[str, List]):
        """应用EWC损失"""
        # 在模型更新中添加EWC惩罚项
        # 这需要修改模型的损失函数
        pass
    
    def _calculate_novelty(self, state: np.ndarray) -> float:
        """计算状态新颖性"""
        if len(self.state_embeddings) == 0:
            self.state_embeddings.append(state)
            return 1.0
        
        # 计算与历史状态的最小距离
        state_array = np.array(list(self.state_embeddings))
        distances = np.linalg.norm(state_array - state, axis=1)
        min_distance = np.min(distances)
        
        # 添加到嵌入集合
        self.state_embeddings.append(state)
        
        # 归一化新颖性分数
        novelty = 1 - np.exp(-min_distance)
        return novelty
    
    def _validate_update(self, updated_model: Any, original_model: Any, 
                       model_type: ModelType) -> bool:
        """验证更新效果"""
        # 使用保留的验证集测试
        validation_memories = list(self.core_memories)[-50:] if len(self.core_memories) >= 50 else list(self.core_memories)
        
        if not validation_memories:
            return True
        
        # 比较新旧模型性能
        original_score = self._evaluate_model(original_model, validation_memories)
        updated_score = self._evaluate_model(updated_model, validation_memories)
        
        # 检查是否有严重退化
        degradation = (original_score - updated_score) / original_score if original_score > 0 else 0
        
        if degradation > 0.2:  # 性能下降超过20%
            self.logger.warning(f"检测到性能退化: {degradation:.2%}")
            
            # 记录遗忘指标
            self.forgetting_metrics[model_type].append({
                'timestamp': TimeUtils.now_timestamp(),
                'degradation': degradation,
                'original_score': original_score,
                'updated_score': updated_score
            })
            
            return False
        
        return True
    
    def _evaluate_model(self, model: Any, memories: List[ExperienceMemory]) -> float:
        """评估模型性能"""
        if not memories:
            return 0.5
        
        correct_predictions = 0
        
        for memory in memories:
            # 简化的评估逻辑
            # 实际应该根据模型类型实现具体的评估方法
            if hasattr(model, 'predict'):
                prediction = model.predict(memory.state)
                if prediction == memory.action:
                    correct_predictions += 1
        
        return correct_predictions / len(memories)
    
    def _update_memories(self, new_memories: List[ExperienceMemory], 
                       model_type: ModelType):
        """更新记忆库"""
        for memory in new_memories:
            # 按市场状态分类存储
            self.episodic_memory[memory.market_state].append(memory)
        
        # 定期清理低重要性记忆
        if len(new_memories) > 100:
            self._cleanup_memories()
    
    def _cleanup_memories(self):
        """清理低重要性记忆"""
        for market_state, memories in self.episodic_memory.items():
            if len(memories) >= memories.maxlen * 0.9:
                # 按重要性排序
                sorted_memories = sorted(memories, key=lambda m: m.importance, reverse=True)
                
                # 保留高重要性记忆
                keep_count = int(memories.maxlen * 0.7)
                self.episodic_memory[market_state] = deque(
                    sorted_memories[:keep_count], 
                    maxlen=memories.maxlen
                )
    
    def _save_model_snapshot(self, model: Any, model_type: ModelType,
                           performance: Dict[str, float]):
        """保存模型快照"""
        try:
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            elif hasattr(model, 'policy_net'):
                state_dict = model.policy_net.state_dict()
            else:
                state_dict = {}
            
            snapshot = ModelSnapshot(
                model_type=model_type,
                timestamp=TimeUtils.now_timestamp(),
                state_dict=copy.deepcopy(state_dict),
                performance_metrics=performance,
                market_conditions=self._get_current_market_conditions(),
                version=f"v{len(self.model_snapshots[model_type])}"
            )
            
            self.model_snapshots[model_type].append(snapshot)
            
            self.logger.info(f"保存模型快照: {model_type.value} - {snapshot.version}")
            
        except Exception as e:
            self.logger.error(f"保存快照失败: {e}")
    
    def _get_current_market_conditions(self) -> Dict[str, Any]:
        """获取当前市场条件"""
        # 简化实现，实际应该从市场环境分类器获取
        return {
            'volatility': 'normal',
            'trend': 'neutral',
            'volume': 'average'
        }
    
    def _record_update(self, model_type: ModelType, performance: Dict[str, float]):
        """记录更新"""
        update_record = {
            'timestamp': TimeUtils.now_timestamp(),
            'model_type': model_type,
            'performance': performance,
            'memories_used': sum(len(m) for m in self.episodic_memory.values()),
            'core_memories': len(self.core_memories)
        }
        
        self.update_history.append(update_record)
    
    def _get_recent_performance(self, model_type: ModelType) -> float:
        """获取最近性能"""
        recent_updates = [u for u in self.update_history 
                         if u['model_type'] == model_type][-10:]
        
        if not recent_updates:
            return 0.5
        
        performances = [u['performance'].get('accuracy', 0.5) for u in recent_updates]
        return np.mean(performances)
    
    def _count_new_samples(self, model_type: ModelType, since_timestamp: int) -> int:
        """计算新样本数"""
        count = 0
        for memories in self.episodic_memory.values():
            for memory in memories:
                if hasattr(memory, 'timestamp') and memory.timestamp > since_timestamp:
                    count += 1
        return count
    
    def rollback_model(self, model_type: ModelType, version: Optional[str] = None) -> Optional[Any]:
        """回滚模型到之前版本"""
        if model_type not in self.model_snapshots:
            return None
        
        snapshots = self.model_snapshots[model_type]
        if not snapshots:
            return None
        
        if version:
            # 查找特定版本
            for snapshot in snapshots:
                if snapshot.version == version:
                    self.logger.info(f"回滚到版本: {version}")
                    return snapshot.state_dict
        else:
            # 回滚到上一个版本
            if len(snapshots) >= 2:
                self.logger.info(f"回滚到上一版本: {snapshots[-2].version}")
                return snapshots[-2].state_dict
        
        return None
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        stats = {
            'total_memories': sum(len(m) for m in self.episodic_memory.values()),
            'core_memories': len(self.core_memories),
            'memories_by_state': {
                state.value: len(memories)
                for state, memories in self.episodic_memory.items()
            },
            'update_count': len(self.update_history),
            'model_versions': {
                model_type.value: len(snapshots)
                for model_type, snapshots in self.model_snapshots.items()
            },
            'forgetting_events': {
                model_type.value: len(events)
                for model_type, events in self.forgetting_metrics.items()
            }
        }
        
        # 计算平均遗忘率
        total_forgetting_events = sum(len(events) for events in self.forgetting_metrics.values())
        if self.update_history:
            stats['forgetting_rate'] = total_forgetting_events / len(self.update_history)
        else:
            stats['forgetting_rate'] = 0
        
        return stats
    
    def save_memory_bank(self, filepath: str):
        """保存记忆库"""
        memory_data = {
            'episodic_memory': {},
            'core_memories': [],
            'state_embeddings': list(self.state_embeddings),
            'statistics': self.get_learning_statistics()
        }
        
        # 序列化记忆
        for state, memories in self.episodic_memory.items():
            memory_data['episodic_memory'][state.value] = [
                {
                    'state': mem.state.tolist(),
                    'action': mem.action,
                    'reward': mem.reward,
                    'next_state': mem.next_state.tolist(),
                    'market_state': mem.market_state.value,
                    'importance': mem.importance,
                    'age': mem.age,
                    'usage_count': mem.usage_count
                }
                for mem in memories
            ]
        
        memory_data['core_memories'] = [
            {
                'state': mem.state.tolist(),
                'action': mem.action,
                'reward': mem.reward,
                'importance': mem.importance
            }
            for mem in self.core_memories
        ]
        
        # 压缩保存
        import gzip
        with gzip.open(filepath + '.gz', 'wt', encoding='utf-8') as f:
            json.dump(memory_data, f)
        
        self.logger.info(f"记忆库已保存: {filepath}.gz")
    
    def load_memory_bank(self, filepath: str):
        """加载记忆库"""
        try:
            import gzip
            with gzip.open(filepath + '.gz', 'rt', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # 恢复记忆
            self.episodic_memory.clear()
            for state_str, memories in memory_data['episodic_memory'].items():
                state = MarketState(state_str)
                for mem_data in memories:
                    memory = ExperienceMemory(
                        state=np.array(mem_data['state']),
                        action=mem_data['action'],
                        reward=mem_data['reward'],
                        next_state=np.array(mem_data['next_state']),
                        market_state=MarketState(mem_data['market_state']),
                        importance=mem_data['importance'],
                        age=mem_data['age'],
                        usage_count=mem_data['usage_count']
                    )
                    self.episodic_memory[state].append(memory)
            
            self.logger.info(f"记忆库已加载: {filepath}.gz")
            
        except Exception as e:
            self.logger.error(f"加载记忆库失败: {e}")