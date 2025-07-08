# system_state_manager.py - 系统状态管理
# =============================================================================
# 核心职责：
# 1. 系统状态持久化
# 2. 断点恢复功能
# 3. 状态版本控制
# 4. 配置热更新
# 5. 状态同步
# =============================================================================

import json
import pickle
import asyncio
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import shutil
import threading
from collections import deque

from logger import TradingLogger
from utils import ConfigManager, TimeUtils


@dataclass
class SystemState:
    """系统状态"""
    version: str = "1.0.0"
    timestamp: int = field(default_factory=TimeUtils.now_timestamp)
    
    # 交易状态
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_orders: List[Dict[str, Any]] = field(default_factory=list)
    trade_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 模型状态
    model_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    feature_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 系统状态
    component_states: Dict[str, str] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    config_version: int = 0
    
    # 市场状态
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    last_prices: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """从字典创建"""
        return cls(**data)


@dataclass
class StateSnapshot:
    """状态快照"""
    id: str
    state: SystemState
    timestamp: int
    description: str = ""
    
    def save(self, path: Path):
        """保存快照"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> 'StateSnapshot':
        """加载快照"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class SystemStateManager:
    """
    系统状态管理器
    
    核心功能：
    1. 实时状态跟踪
    2. 定期状态保存
    3. 快照管理
    4. 状态恢复
    5. 配置同步
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("STATE_MANAGER")
        
        # 状态存储
        self.current_state = SystemState()
        self.state_lock = threading.RLock()
        
        # 存储路径
        self.state_dir = Path(self.config.get("state.directory", "states"))
        self.state_dir.mkdir(exist_ok=True)
        
        self.state_file = self.state_dir / "current_state.json"
        self.snapshot_dir = self.state_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # 配置
        self.auto_save_interval = self.config.get("state.auto_save_interval", 300)  # 5分钟
        self.max_snapshots = self.config.get("state.max_snapshots", 100)
        self.enable_compression = self.config.get("state.enable_compression", True)
        
        # 状态变更历史
        self.state_history: deque = deque(maxlen=1000)
        
        # 订阅者
        self.state_subscribers: Set[Callable] = set()
        
        # 自动保存任务
        self.auto_save_task = None
        self.is_running = False
        
        self.logger.info("系统状态管理器初始化完成")
    
    async def initialize(self):
        """初始化"""
        # 加载最新状态
        await self.restore_state()
        
        # 启动自动保存
        self.is_running = True
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        
        self.logger.info("状态管理器已启动")
    
    async def save_state(self, partial_state: Optional[Dict[str, Any]] = None) -> bool:
        """保存状态"""
        try:
            with self.state_lock:
                # 更新部分状态
                if partial_state:
                    self._update_partial_state(partial_state)
                
                # 更新时间戳
                self.current_state.timestamp = TimeUtils.now_timestamp()
                
                # 保存到文件
                state_dict = self.current_state.to_dict()
                
                # 写入临时文件
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)
                
                # 原子性替换
                temp_file.replace(self.state_file)
                
                # 记录历史
                self._record_state_change('save', partial_state)
                
                # 通知订阅者
                await self._notify_subscribers('state_saved', state_dict)
                
                self.logger.debug("状态已保存")
                return True
                
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
            return False
    
    async def restore_state(self) -> bool:
        """恢复状态"""
        try:
            if not self.state_file.exists():
                self.logger.info("无历史状态文件")
                return False
            
            # 读取状态文件
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            # 恢复状态
            self.current_state = SystemState.from_dict(state_dict)
            
            # 验证状态
            if not self._validate_state(self.current_state):
                self.logger.warning("状态验证失败，使用默认状态")
                self.current_state = SystemState()
                return False
            
            # 记录恢复
            self._record_state_change('restore', state_dict)
            
            # 通知订阅者
            await self._notify_subscribers('state_restored', state_dict)
            
            self.logger.info(
                f"成功恢复状态，版本: {self.current_state.version}, "
                f"时间: {TimeUtils.timestamp_to_str(self.current_state.timestamp)}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"恢复状态失败: {e}")
            return False
    
    async def create_snapshot(self, description: str = "") -> str:
        """创建快照"""
        try:
            # 生成快照ID
            snapshot_id = f"snapshot_{TimeUtils.now_timestamp()}"
            
            # 创建快照
            snapshot = StateSnapshot(
                id=snapshot_id,
                state=self.current_state,
                timestamp=TimeUtils.now_timestamp(),
                description=description
            )
            
            # 保存快照
            snapshot_file = self.snapshot_dir / f"{snapshot_id}.pkl"
            snapshot.save(snapshot_file)
            
            # 清理旧快照
            self._cleanup_old_snapshots()
            
            # 记录
            self._record_state_change('snapshot', {'id': snapshot_id, 'description': description})
            
            self.logger.info(f"创建快照: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"创建快照失败: {e}")
            return ""
    
    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """恢复快照"""
        try:
            snapshot_file = self.snapshot_dir / f"{snapshot_id}.pkl"
            
            if not snapshot_file.exists():
                self.logger.error(f"快照不存在: {snapshot_id}")
                return False
            
            # 加载快照
            snapshot = StateSnapshot.load(snapshot_file)
            
            # 备份当前状态
            await self.create_snapshot("auto_backup_before_restore")
            
            # 恢复状态
            with self.state_lock:
                self.current_state = snapshot.state
            
            # 保存到文件
            await self.save_state()
            
            # 记录
            self._record_state_change('restore_snapshot', {'id': snapshot_id})
            
            # 通知订阅者
            await self._notify_subscribers('snapshot_restored', {'snapshot_id': snapshot_id})
            
            self.logger.info(f"成功恢复快照: {snapshot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复快照失败: {e}")
            return False
    
    def get_state(self, key: Optional[str] = None) -> Any:
        """获取状态"""
        with self.state_lock:
            if key is None:
                return self.current_state.to_dict()
            
            # 支持点分隔的路径
            keys = key.split('.')
            value = self.current_state.to_dict()
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            
            return value
    
    def update_state(self, key: str, value: Any):
        """更新状态"""
        with self.state_lock:
            # 支持点分隔的路径
            keys = key.split('.')
            
            # 获取父对象
            obj = self.current_state
            for k in keys[:-1]:
                obj = getattr(obj, k)
            
            # 设置值
            setattr(obj, keys[-1], value)
            
            # 记录变更
            self._record_state_change('update', {key: value})
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """更新持仓"""
        with self.state_lock:
            self.current_state.positions[symbol] = position_data
    
    def update_trade_stats(self, stats: Dict[str, Any]):
        """更新交易统计"""
        with self.state_lock:
            self.current_state.trade_stats.update(stats)
    
    def update_model_state(self, model_name: str, state: Dict[str, Any]):
        """更新模型状态"""
        with self.state_lock:
            self.current_state.model_states[model_name] = state
    
    def update_component_state(self, component: str, state: str):
        """更新组件状态"""
        with self.state_lock:
            self.current_state.component_states[component] = state
    
    def increment_error_count(self, component: str):
        """增加错误计数"""
        with self.state_lock:
            if component not in self.current_state.error_counts:
                self.current_state.error_counts[component] = 0
            self.current_state.error_counts[component] += 1
    
    def reset_error_count(self, component: str):
        """重置错误计数"""
        with self.state_lock:
            if component in self.current_state.error_counts:
                self.current_state.error_counts[component] = 0
    
    def _update_partial_state(self, partial_state: Dict[str, Any]):
        """更新部分状态"""
        for key, value in partial_state.items():
            if hasattr(self.current_state, key):
                setattr(self.current_state, key, value)
    
    def _validate_state(self, state: SystemState) -> bool:
        """验证状态有效性"""
        try:
            # 检查版本
            if not state.version:
                return False
            
            # 检查时间戳
            if state.timestamp <= 0:
                return False
            
            # 检查时间戳不是未来时间
            if state.timestamp > TimeUtils.now_timestamp() + 60000:  # 允许1分钟误差
                return False
            
            return True
            
        except Exception:
            return False
    
    def _cleanup_old_snapshots(self):
        """清理旧快照"""
        try:
            # 获取所有快照文件
            snapshots = list(self.snapshot_dir.glob("snapshot_*.pkl"))
            
            # 按时间排序
            snapshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除超出限制的快照
            for snapshot in snapshots[self.max_snapshots:]:
                snapshot.unlink()
                self.logger.debug(f"删除旧快照: {snapshot.name}")
                
        except Exception as e:
            self.logger.error(f"清理快照失败: {e}")
    
    def _record_state_change(self, action: str, data: Any):
        """记录状态变更"""
        record = {
            'timestamp': TimeUtils.now_timestamp(),
            'action': action,
            'data': data
        }
        self.state_history.append(record)
    
    async def _notify_subscribers(self, event: str, data: Any):
        """通知订阅者"""
        for subscriber in self.state_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event, data)
                else:
                    subscriber(event, data)
            except Exception as e:
                self.logger.error(f"通知订阅者失败: {e}")
    
    async def _auto_save_loop(self):
        """自动保存循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                await self.save_state()
                
            except Exception as e:
                self.logger.error(f"自动保存失败: {e}")
    
    def subscribe(self, callback: Callable):
        """订阅状态变更"""
        self.state_subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """取消订阅"""
        self.state_subscribers.discard(callback)
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """获取快照列表"""
        snapshots = []
        
        try:
            for snapshot_file in self.snapshot_dir.glob("snapshot_*.pkl"):
                try:
                    snapshot = StateSnapshot.load(snapshot_file)
                    snapshots.append({
                        'id': snapshot.id,
                        'timestamp': snapshot.timestamp,
                        'description': snapshot.description,
                        'file': snapshot_file.name
                    })
                except Exception:
                    continue
            
            # 按时间倒序
            snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"获取快照列表失败: {e}")
        
        return snapshots
    
    def export_state(self, export_path: Path) -> bool:
        """导出状态"""
        try:
            # 创建导出目录
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 复制当前状态
            shutil.copy2(self.state_file, export_path / "current_state.json")
            
            # 复制配置
            config_file = Path(self.config.config_path)
            if config_file.exists():
                shutil.copy2(config_file, export_path / "config.yaml")
            
            # 创建信息文件
            info = {
                'export_time': TimeUtils.now_timestamp(),
                'version': self.current_state.version,
                'state_timestamp': self.current_state.timestamp
            }
            
            with open(export_path / "export_info.json", 'w') as f:
                json.dump(info, f, indent=2)
            
            self.logger.info(f"状态已导出到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出状态失败: {e}")
            return False
    
    def import_state(self, import_path: Path) -> bool:
        """导入状态"""
        try:
            # 检查文件
            state_file = import_path / "current_state.json"
            if not state_file.exists():
                self.logger.error("导入路径中无状态文件")
                return False
            
            # 备份当前状态
            asyncio.create_task(self.create_snapshot("auto_backup_before_import"))
            
            # 导入状态
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            self.current_state = SystemState.from_dict(state_dict)
            
            # 导入配置（可选）
            config_file = import_path / "config.yaml"
            if config_file.exists():
                shutil.copy2(config_file, self.config.config_path)
                self.config.load_config()
            
            # 保存状态
            asyncio.create_task(self.save_state())
            
            self.logger.info(f"成功导入状态: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导入状态失败: {e}")
            return False
    
    async def shutdown(self):
        """关闭状态管理器"""
        self.logger.info("正在关闭状态管理器...")
        
        # 停止自动保存
        self.is_running = False
        
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
        
        # 最终保存
        await self.save_state()
        
        self.logger.info("状态管理器已关闭")

