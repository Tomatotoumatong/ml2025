# lock_manager.py - 进程锁管理模块
# =============================================================================
# 核心职责：
# 1. 防止多实例运行
# 2. 资源互斥访问控制
# 3. 分布式锁支持
# 4. 死锁检测和恢复
# 5. 锁状态监控
# =============================================================================

import os
import sys
import time
import fcntl
import socket
import threading
import psutil
from typing import Dict, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import hashlib
from contextlib import contextmanager

from logger import TradingLogger
from utils import ConfigManager, TimeUtils


class LockType(Enum):
    """锁类型枚举"""
    PROCESS = "process"      # 进程锁
    RESOURCE = "resource"    # 资源锁
    OPERATION = "operation"  # 操作锁
    DISTRIBUTED = "distributed"  # 分布式锁


class LockState(Enum):
    """锁状态枚举"""
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    WAITING = "waiting"
    EXPIRED = "expired"


@dataclass
class LockInfo:
    """锁信息"""
    name: str
    type: LockType
    holder: str  # 持有者标识
    acquired_at: int
    expires_at: Optional[int]
    pid: Optional[int]
    host: str
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """检查锁是否过期"""
        if self.expires_at is None:
            return False
        return TimeUtils.now_timestamp() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type.value,
            'holder': self.holder,
            'acquired_at': self.acquired_at,
            'expires_at': self.expires_at,
            'pid': self.pid,
            'host': self.host,
            'metadata': self.metadata
        }


class ProcessLock:
    """进程级锁实现"""
    
    def __init__(self, lockfile: str):
        self.lockfile = lockfile
        self.fd = None
        self.logger = TradingLogger().get_logger("PROCESS_LOCK")
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取锁"""
        try:
            self.fd = open(self.lockfile, 'w')
            
            if blocking:
                if timeout:
                    # 带超时的阻塞获取
                    start_time = time.time()
                    while True:
                        try:
                            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except IOError:
                            if time.time() - start_time > timeout:
                                self.fd.close()
                                self.fd = None
                                return False
                            time.sleep(0.1)
                else:
                    # 无限阻塞
                    fcntl.flock(self.fd, fcntl.LOCK_EX)
            else:
                # 非阻塞
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # 写入进程信息
            self.fd.write(f"{os.getpid()}\n")
            self.fd.flush()
            
            return True
            
        except IOError:
            if self.fd:
                self.fd.close()
                self.fd = None
            return False
    
    def release(self):
        """释放锁"""
        if self.fd:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
                self.fd = None
                
                # 删除锁文件
                try:
                    os.unlink(self.lockfile)
                except:
                    pass
                    
            except Exception as e:
                self.logger.error(f"释放锁失败: {e}")
    
    def is_locked(self) -> bool:
        """检查是否已锁定"""
        try:
            with open(self.lockfile, 'w') as fd:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
        except IOError:
            return True
        except FileNotFoundError:
            return False


class ResourceLock:
    """资源级锁实现（基于线程锁）"""
    
    def __init__(self, name: str, timeout: Optional[float] = None):
        self.name = name
        self.timeout = timeout
        self.lock = threading.RLock()
        self.holder = None
        self.acquired_at = None
        self.count = 0
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取锁"""
        timeout = timeout or self.timeout
        
        acquired = self.lock.acquire(blocking, timeout or -1)
        
        if acquired:
            self.holder = threading.current_thread().ident
            self.acquired_at = TimeUtils.now_timestamp()
            self.count += 1
        
        return acquired
    
    def release(self):
        """释放锁"""
        if self.holder == threading.current_thread().ident:
            self.count -= 1
            if self.count == 0:
                self.holder = None
                self.acquired_at = None
            self.lock.release()
    
    def is_locked(self) -> bool:
        """检查是否已锁定"""
        return self.holder is not None


class LockManager:
    """
    锁管理器
    
    核心功能：
    1. 统一的锁管理接口
    2. 死锁检测
    3. 锁超时处理
    4. 分布式锁支持
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("LOCK_MANAGER")
        
        # 锁存储
        self.locks: Dict[str, Any] = {}
        self.lock_info: Dict[str, LockInfo] = {}
        
        # 配置
        self.lock_dir = Path(self.config.get("lock.directory", "/tmp/stark4_locks"))
        self.lock_dir.mkdir(exist_ok=True)
        
        self.default_timeout = self.config.get("lock.default_timeout", 300)  # 5分钟
        self.check_interval = self.config.get("lock.check_interval", 30)  # 30秒
        
        # 死锁检测
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.waiting_graph: Dict[str, str] = {}  # thread_id -> lock_name
        
        # 监控线程
        self.monitor_thread = None
        self.is_running = False
        
        # 主进程锁
        self.process_lock = None
        
        self.logger.info("锁管理器初始化完成")
    
    def ensure_single_instance(self, app_name: str = "stark4") -> bool:
        """确保单实例运行"""
        lockfile = self.lock_dir / f"{app_name}.pid"
        
        # 检查是否已有实例在运行
        if lockfile.exists():
            try:
                with open(lockfile, 'r') as f:
                    pid = int(f.read().strip())
                
                # 检查进程是否存在
                if psutil.pid_exists(pid):
                    try:
                        process = psutil.Process(pid)
                        if app_name in process.name() or app_name in ' '.join(process.cmdline()):
                            self.logger.error(f"已有实例在运行 (PID: {pid})")
                            return False
                    except:
                        pass
                
                # 进程不存在，删除旧锁文件
                lockfile.unlink()
                
            except Exception as e:
                self.logger.warning(f"检查锁文件失败: {e}")
        
        # 创建进程锁
        self.process_lock = ProcessLock(str(lockfile))
        
        if not self.process_lock.acquire(blocking=False):
            self.logger.error("无法获取进程锁")
            return False
        
        self.logger.info(f"进程锁获取成功 (PID: {os.getpid()})")
        return True
    
    def acquire_lock(self, name: str, lock_type: LockType = LockType.RESOURCE,
                    blocking: bool = True, timeout: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        获取锁
        
        Args:
            name: 锁名称
            lock_type: 锁类型
            blocking: 是否阻塞
            timeout: 超时时间
            metadata: 元数据
        
        Returns:
            是否成功获取锁
        """
        try:
            # 检查是否已存在
            if name in self.locks:
                lock = self.locks[name]
                if lock.is_locked():
                    if not blocking:
                        return False
                    # 等待锁释放
                    if not self._wait_for_lock(name, timeout):
                        return False
            
            # 创建锁
            if lock_type == LockType.PROCESS:
                lockfile = self.lock_dir / f"{name}.lock"
                lock = ProcessLock(str(lockfile))
            elif lock_type == LockType.RESOURCE:
                lock = ResourceLock(name, timeout)
            elif lock_type == LockType.DISTRIBUTED:
                # TODO: 实现分布式锁
                lock = ResourceLock(name, timeout)
            else:
                lock = ResourceLock(name, timeout)
            
            # 死锁检测
            if blocking and not self._check_deadlock(name):
                self.logger.error(f"检测到潜在死锁: {name}")
                return False
            
            # 获取锁
            if lock.acquire(blocking, timeout):
                self.locks[name] = lock
                
                # 记录锁信息
                self.lock_info[name] = LockInfo(
                    name=name,
                    type=lock_type,
                    holder=self._get_holder_id(),
                    acquired_at=TimeUtils.now_timestamp(),
                    expires_at=TimeUtils.now_timestamp() + (timeout or self.default_timeout) * 1000,
                    pid=os.getpid(),
                    host=socket.gethostname(),
                    metadata=metadata or {}
                )
                
                self.logger.debug(f"锁获取成功: {name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"获取锁失败 {name}: {e}")
            return False
    
    def release_lock(self, name: str) -> bool:
        """释放锁"""
        try:
            if name not in self.locks:
                self.logger.warning(f"锁不存在: {name}")
                return False
            
            lock = self.locks[name]
            lock.release()
            
            # 清理记录
            del self.locks[name]
            if name in self.lock_info:
                del self.lock_info[name]
            
            # 清理依赖图
            if name in self.dependency_graph:
                del self.dependency_graph[name]
            
            self.logger.debug(f"锁释放成功: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"释放锁失败 {name}: {e}")
            return False
    
    def is_locked(self, name: str) -> bool:
        """检查锁状态"""
        if name in self.locks:
            lock = self.locks[name]
            return lock.is_locked()
        return False
    
    @contextmanager
    def lock_context(self, name: str, lock_type: LockType = LockType.RESOURCE,
                    timeout: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """锁上下文管理器"""
        acquired = False
        try:
            acquired = self.acquire_lock(name, lock_type, blocking=True, 
                                       timeout=timeout, metadata=metadata)
            if not acquired:
                raise RuntimeError(f"无法获取锁: {name}")
            yield
        finally:
            if acquired:
                self.release_lock(name)
    
    def _get_holder_id(self) -> str:
        """获取持有者标识"""
        thread_id = threading.current_thread().ident
        process_id = os.getpid()
        return f"{process_id}:{thread_id}"
    
    def _wait_for_lock(self, name: str, timeout: Optional[float]) -> bool:
        """等待锁释放"""
        start_time = time.time()
        timeout = timeout or self.default_timeout
        
        # 记录等待关系（用于死锁检测）
        holder_id = self._get_holder_id()
        self.waiting_graph[holder_id] = name
        
        try:
            while True:
                if not self.is_locked(name):
                    return True
                
                if timeout and (time.time() - start_time) > timeout:
                    return False
                
                time.sleep(0.1)
        finally:
            # 清理等待关系
            if holder_id in self.waiting_graph:
                del self.waiting_graph[holder_id]
    
    def _check_deadlock(self, requested_lock: str) -> bool:
        """
        检查死锁
        
        使用等待图检测循环依赖
        """
        holder_id = self._get_holder_id()
        
        # 构建当前的等待关系
        waiting_for = {}
        lock_holders = {}
        
        # 收集锁持有者信息
        for lock_name, info in self.lock_info.items():
            lock_holders[lock_name] = info.holder
        
        # 收集等待关系
        for waiter, lock_name in self.waiting_graph.items():
            if lock_name in lock_holders:
                waiting_for[waiter] = lock_holders[lock_name]
        
        # 添加当前请求
        if requested_lock in lock_holders:
            waiting_for[holder_id] = lock_holders[requested_lock]
        
        # 检测循环
        visited = set()
        
        def has_cycle(node: str) -> bool:
            if node in visited:
                return True
            visited.add(node)
            
            if node in waiting_for:
                return has_cycle(waiting_for[node])
            
            return False
        
        return not has_cycle(holder_id)
    
    def start_monitor(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("锁监控已启动")
    
    def stop_monitor(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("锁监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检查过期锁
                self._check_expired_locks()
                
                # 检查死锁
                self._detect_deadlocks()
                
                # 清理无效锁
                self._cleanup_invalid_locks()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"锁监控异常: {e}")
    
    def _check_expired_locks(self):
        """检查过期锁"""
        expired_locks = []
        
        for name, info in self.lock_info.items():
            if info.is_expired():
                expired_locks.append(name)
                self.logger.warning(f"检测到过期锁: {name}")
        
        # 强制释放过期锁
        for name in expired_locks:
            try:
                self.release_lock(name)
                self.logger.info(f"已释放过期锁: {name}")
            except Exception as e:
                self.logger.error(f"释放过期锁失败 {name}: {e}")
    
    def _detect_deadlocks(self):
        """检测死锁"""
        # 构建等待图
        waiting_chains = {}
        
        for waiter, lock_name in self.waiting_graph.items():
            if lock_name in self.lock_info:
                holder = self.lock_info[lock_name].holder
                waiting_chains[waiter] = holder
        
        # 查找循环
        for start_node in waiting_chains:
            visited = set()
            current = start_node
            
            while current in waiting_chains:
                if current in visited:
                    # 发现循环
                    self.logger.error(f"检测到死锁循环: {visited}")
                    # TODO: 实现死锁恢复策略
                    break
                
                visited.add(current)
                current = waiting_chains[current]
    
    def _cleanup_invalid_locks(self):
        """清理无效锁"""
        invalid_locks = []
        
        for name, info in self.lock_info.items():
            # 检查进程是否存在
            if info.pid and not psutil.pid_exists(info.pid):
                invalid_locks.append(name)
                self.logger.warning(f"检测到无效锁（进程不存在）: {name}")
        
        # 清理无效锁
        for name in invalid_locks:
            try:
                if name in self.locks:
                    # 强制释放
                    del self.locks[name]
                    del self.lock_info[name]
                    
                    # 如果是文件锁，删除文件
                    lockfile = self.lock_dir / f"{name}.lock"
                    if lockfile.exists():
                        lockfile.unlink()
                    
                    self.logger.info(f"已清理无效锁: {name}")
                    
            except Exception as e:
                self.logger.error(f"清理无效锁失败 {name}: {e}")
    
    def get_lock_info(self, name: str) -> Optional[LockInfo]:
        """获取锁信息"""
        return self.lock_info.get(name)
    
    def get_all_locks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有锁信息"""
        return {
            name: info.to_dict()
            for name, info in self.lock_info.items()
        }
    
    def force_release_lock(self, name: str):
        """强制释放锁（管理员功能）"""
        self.logger.warning(f"强制释放锁: {name}")
        
        try:
            # 删除锁记录
            if name in self.locks:
                del self.locks[name]
            if name in self.lock_info:
                del self.lock_info[name]
            
            # 删除锁文件
            lockfile = self.lock_dir / f"{name}.lock"
            if lockfile.exists():
                lockfile.unlink()
            
            self.logger.info(f"锁已强制释放: {name}")
            
        except Exception as e:
            self.logger.error(f"强制释放锁失败 {name}: {e}")
    
    def cleanup(self):
        """清理所有锁"""
        self.logger.info("清理所有锁...")
        
        # 停止监控
        self.stop_monitor()
        
        # 释放所有锁
        for name in list(self.locks.keys()):
            try:
                self.release_lock(name)
            except:
                pass
        
        # 释放进程锁
        if self.process_lock:
            self.process_lock.release()
        
        self.logger.info("锁清理完成")


# 全局锁管理器实例
_lock_manager = None


def get_lock_manager(config_path: str = "config.yaml") -> LockManager:
    """获取全局锁管理器实例"""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = LockManager(config_path)
    return _lock_manager