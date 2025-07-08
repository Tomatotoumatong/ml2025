# rate_limit_manager.py - API限流模块
# =============================================================================
# 核心职责：
# 1. API请求频率限制
# 2. 令牌桶算法实现
# 3. 请求队列管理
# 4. 多交易所限流协调
# 5. 限流统计和监控
# =============================================================================

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import heapq

from logger import TradingLogger
from utils import ConfigManager, TimeUtils


class RateLimitType(Enum):
    """限流类型枚举"""
    TOKEN_BUCKET = "token_bucket"      # 令牌桶
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口
    FIXED_WINDOW = "fixed_window"      # 固定窗口
    LEAKY_BUCKET = "leaky_bucket"      # 漏桶


class RequestPriority(Enum):
    """请求优先级枚举"""
    CRITICAL = 1    # 关键请求（如止损单）
    HIGH = 2        # 高优先级（如交易请求）
    NORMAL = 3      # 普通优先级（如查询请求）
    LOW = 4         # 低优先级（如历史数据）


@dataclass
class RateLimitConfig:
    """限流配置"""
    name: str
    type: RateLimitType
    capacity: int  # 容量（请求数或令牌数）
    refill_rate: float  # 填充速率（每秒）
    window_size: int = 60  # 窗口大小（秒）
    burst_size: Optional[int] = None  # 突发容量


@dataclass
class Request:
    """请求信息"""
    id: str
    endpoint: str
    priority: RequestPriority
    timestamp: int
    weight: int = 1  # 请求权重（消耗令牌数）
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.priority.value < other.priority.value


@dataclass
class RateLimitStats:
    """限流统计"""
    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    total_wait_time: float = 0
    last_reset: int = field(default_factory=TimeUtils.now_timestamp)
    
    def acceptance_rate(self) -> float:
        """计算接受率"""
        if self.total_requests == 0:
            return 1.0
        return self.accepted_requests / self.total_requests


class TokenBucket:
    """令牌桶实现"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self):
        """填充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # 计算应该添加的令牌数
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """消费令牌"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def available_tokens(self) -> float:
        """获取可用令牌数"""
        with self.lock:
            self._refill()
            return self.tokens
    
    def time_until_tokens(self, tokens: int) -> float:
        """计算获得指定令牌数需要的时间"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0
            
            needed = tokens - self.tokens
            return needed / self.refill_rate


class SlidingWindowCounter:
    """滑动窗口计数器"""
    
    def __init__(self, window_size: int, capacity: int):
        self.window_size = window_size * 1000  # 转换为毫秒
        self.capacity = capacity
        self.requests = deque()
        self.lock = threading.Lock()
    
    def add_request(self, weight: int = 1) -> bool:
        """添加请求"""
        with self.lock:
            now = TimeUtils.now_timestamp()
            
            # 清理过期请求
            while self.requests and self.requests[0][0] < now - self.window_size:
                self.requests.popleft()
            
            # 计算当前窗口内的请求数
            current_count = sum(req[1] for req in self.requests)
            
            if current_count + weight <= self.capacity:
                self.requests.append((now, weight))
                return True
            
            return False
    
    def current_count(self) -> int:
        """获取当前计数"""
        with self.lock:
            now = TimeUtils.now_timestamp()
            
            # 清理过期请求
            while self.requests and self.requests[0][0] < now - self.window_size:
                self.requests.popleft()
            
            return sum(req[1] for req in self.requests)


class RateLimiter:
    """限流器基类"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.stats = RateLimitStats()
        
        # 根据类型创建具体实现
        if config.type == RateLimitType.TOKEN_BUCKET:
            self.impl = TokenBucket(config.capacity, config.refill_rate)
        elif config.type == RateLimitType.SLIDING_WINDOW:
            self.impl = SlidingWindowCounter(config.window_size, config.capacity)
        else:
            raise ValueError(f"不支持的限流类型: {config.type}")
    
    def allow_request(self, weight: int = 1) -> bool:
        """检查是否允许请求"""
        self.stats.total_requests += 1
        
        if isinstance(self.impl, TokenBucket):
            allowed = self.impl.consume(weight)
        elif isinstance(self.impl, SlidingWindowCounter):
            allowed = self.impl.add_request(weight)
        else:
            allowed = False
        
        if allowed:
            self.stats.accepted_requests += 1
        else:
            self.stats.rejected_requests += 1
        
        return allowed
    
    def wait_time(self, weight: int = 1) -> float:
        """获取需要等待的时间"""
        if isinstance(self.impl, TokenBucket):
            return self.impl.time_until_tokens(weight)
        else:
            # 滑动窗口没有明确的等待时间
            return 0 if self.allow_request(weight) else 1.0


class RateLimitManager:
    """
    限流管理器
    
    核心功能：
    1. 多交易所限流管理
    2. 请求优先级队列
    3. 自适应限流
    4. 限流统计
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("RATE_LIMIT")
        
        # 限流器存储
        self.limiters: Dict[str, RateLimiter] = {}
        self._init_limiters()
        
        # 请求队列
        self.request_queues: Dict[str, List[Request]] = defaultdict(list)
        self.queue_lock = threading.Lock()
        
        # 配置
        self.max_queue_size = self.config.get("rate_limit.max_queue_size", 1000)
        self.queue_timeout = self.config.get("rate_limit.queue_timeout", 30)
        
        # 统计
        self.endpoint_stats: Dict[str, RateLimitStats] = defaultdict(RateLimitStats)
        
        # 处理线程
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        
        # 回调函数
        self.limit_exceeded_callbacks: List[Callable] = []
        
        self.logger.info("限流管理器初始化完成")
    
    def _init_limiters(self):
        """初始化限流器"""
        # Binance限流配置
        self.limiters['binance_weight'] = RateLimiter(RateLimitConfig(
            name='binance_weight',
            type=RateLimitType.SLIDING_WINDOW,
            capacity=self.config.get("rate_limit.binance.weight_limit", 1200),
            refill_rate=0,
            window_size=60
        ))
        
        self.limiters['binance_order'] = RateLimiter(RateLimitConfig(
            name='binance_order',
            type=RateLimitType.SLIDING_WINDOW,
            capacity=self.config.get("rate_limit.binance.order_limit", 50),
            refill_rate=0,
            window_size=10
        ))
        
        # OKEx限流配置
        self.limiters['okex_public'] = RateLimiter(RateLimitConfig(
            name='okex_public',
            type=RateLimitType.TOKEN_BUCKET,
            capacity=self.config.get("rate_limit.okex.public_limit", 20),
            refill_rate=self.config.get("rate_limit.okex.public_rate", 2)
        ))
        
        self.limiters['okex_private'] = RateLimiter(RateLimitConfig(
            name='okex_private',
            type=RateLimitType.TOKEN_BUCKET,
            capacity=self.config.get("rate_limit.okex.private_limit", 10),
            refill_rate=self.config.get("rate_limit.okex.private_rate", 1)
        ))
    
    def start(self):
        """启动处理线程"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 为每个限流器启动处理线程
        for name in self.limiters:
            thread = threading.Thread(
                target=self._process_queue,
                args=(name,),
                daemon=True
            )
            thread.start()
            self.processing_threads[name] = thread
        
        self.logger.info("限流处理线程已启动")
    
    def stop(self):
        """停止处理线程"""
        self.is_running = False
        
        # 等待所有线程结束
        for thread in self.processing_threads.values():
            thread.join(timeout=5)
        
        self.processing_threads.clear()
        self.logger.info("限流处理线程已停止")
    
    def _process_queue(self, limiter_name: str):
        """处理请求队列"""
        while self.is_running:
            try:
                # 获取队列中的请求
                request = self._get_next_request(limiter_name)
                
                if request is None:
                    time.sleep(0.1)
                    continue
                
                # 检查请求是否超时
                if self._is_request_timeout(request):
                    self.logger.warning(f"请求超时: {request.id}")
                    if request.callback:
                        request.callback(False, "请求超时")
                    continue
                
                # 等待限流许可
                limiter = self.limiters[limiter_name]
                
                while not limiter.allow_request(request.weight):
                    wait_time = limiter.wait_time(request.weight)
                    if wait_time > 0:
                        time.sleep(min(wait_time, 1.0))
                    
                    # 再次检查超时
                    if self._is_request_timeout(request):
                        self.logger.warning(f"等待限流时超时: {request.id}")
                        if request.callback:
                            request.callback(False, "等待限流超时")
                        break
                else:
                    # 请求通过限流
                    self.logger.debug(f"请求通过限流: {request.id}")
                    if request.callback:
                        request.callback(True, None)
                    
                    # 更新统计
                    self._update_endpoint_stats(request.endpoint, True)
                    
            except Exception as e:
                self.logger.error(f"处理队列异常 {limiter_name}: {e}")
    
    def _get_next_request(self, limiter_name: str) -> Optional[Request]:
        """获取下一个请求"""
        with self.queue_lock:
            queue = self.request_queues.get(limiter_name, [])
            if queue:
                # 使用堆实现优先队列
                return heapq.heappop(queue)
        return None
    
    def _is_request_timeout(self, request: Request) -> bool:
        """检查请求是否超时"""
        elapsed = (TimeUtils.now_timestamp() - request.timestamp) / 1000
        return elapsed > self.queue_timeout
    
    def _update_endpoint_stats(self, endpoint: str, success: bool):
        """更新端点统计"""
        stats = self.endpoint_stats[endpoint]
        stats.total_requests += 1
        if success:
            stats.accepted_requests += 1
        else:
            stats.rejected_requests += 1
    
    async def acquire(self, endpoint: str, limiter_name: str,
                     weight: int = 1, priority: RequestPriority = RequestPriority.NORMAL,
                     timeout: Optional[float] = None) -> bool:
        """
        异步获取限流许可
        
        Args:
            endpoint: 请求端点
            limiter_name: 限流器名称
            weight: 请求权重
            priority: 请求优先级
            timeout: 超时时间
        
        Returns:
            是否获得许可
        """
        if limiter_name not in self.limiters:
            self.logger.error(f"未知的限流器: {limiter_name}")
            return False
        
        limiter = self.limiters[limiter_name]
        
        # 尝试立即获取
        if limiter.allow_request(weight):
            self._update_endpoint_stats(endpoint, True)
            return True
        
        # 检查队列是否已满
        with self.queue_lock:
            queue_size = len(self.request_queues[limiter_name])
            if queue_size >= self.max_queue_size:
                self.logger.warning(f"请求队列已满: {limiter_name}")
                self._update_endpoint_stats(endpoint, False)
                
                # 触发回调
                for callback in self.limit_exceeded_callbacks:
                    callback(limiter_name, endpoint)
                
                return False
        
        # 创建异步事件
        event = asyncio.Event()
        result = {'success': False, 'error': None}
        
        def callback(success: bool, error: Optional[str]):
            result['success'] = success
            result['error'] = error
            event.set()
        
        # 创建请求并加入队列
        request = Request(
            id=f"{endpoint}_{TimeUtils.now_timestamp()}",
            endpoint=endpoint,
            priority=priority,
            timestamp=TimeUtils.now_timestamp(),
            weight=weight,
            callback=callback
        )
        
        with self.queue_lock:
            heapq.heappush(self.request_queues[limiter_name], request)
            self.endpoint_stats[endpoint].queued_requests += 1
        
        # 等待结果
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout or self.queue_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"等待限流超时: {endpoint}")
            return False
        
        return result['success']
    
    def check_limit(self, limiter_name: str, weight: int = 1) -> bool:
        """
        同步检查限流（不等待）
        
        Args:
            limiter_name: 限流器名称
            weight: 请求权重
        
        Returns:
            是否有足够配额
        """
        if limiter_name not in self.limiters:
            return False
        
        limiter = self.limiters[limiter_name]
        
        # 只检查，不消耗
        if isinstance(limiter.impl, TokenBucket):
            return limiter.impl.available_tokens() >= weight
        elif isinstance(limiter.impl, SlidingWindowCounter):
            return limiter.impl.current_count() + weight <= limiter.config.capacity
        
        return False
    
    def get_limiter_status(self, limiter_name: str) -> Dict[str, Any]:
        """获取限流器状态"""
        if limiter_name not in self.limiters:
            return {}
        
        limiter = self.limiters[limiter_name]
        stats = limiter.stats
        
        status = {
            'name': limiter_name,
            'type': limiter.config.type.value,
            'capacity': limiter.config.capacity,
            'stats': {
                'total_requests': stats.total_requests,
                'accepted_requests': stats.accepted_requests,
                'rejected_requests': stats.rejected_requests,
                'acceptance_rate': stats.acceptance_rate()
            }
        }
        
        # 添加具体状态
        if isinstance(limiter.impl, TokenBucket):
            status['available_tokens'] = limiter.impl.available_tokens()
            status['refill_rate'] = limiter.config.refill_rate
        elif isinstance(limiter.impl, SlidingWindowCounter):
            status['current_count'] = limiter.impl.current_count()
            status['window_size'] = limiter.config.window_size
        
        # 队列状态
        with self.queue_lock:
            status['queue_size'] = len(self.request_queues[limiter_name])
        
        return status
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """获取端点统计"""
        stats = self.endpoint_stats[endpoint]
        
        return {
            'endpoint': endpoint,
            'total_requests': stats.total_requests,
            'accepted_requests': stats.accepted_requests,
            'rejected_requests': stats.rejected_requests,
            'queued_requests': stats.queued_requests,
            'acceptance_rate': stats.acceptance_rate()
        }
    
    def reset_stats(self):
        """重置统计"""
        for limiter in self.limiters.values():
            limiter.stats = RateLimitStats()
        
        self.endpoint_stats.clear()
        self.logger.info("限流统计已重置")
    
    def add_limit_exceeded_callback(self, callback: Callable[[str, str], None]):
        """添加限流超限回调"""
        self.limit_exceeded_callbacks.append(callback)
    
    def update_limit(self, limiter_name: str, capacity: Optional[int] = None,
                    refill_rate: Optional[float] = None):
        """动态更新限流配置"""
        if limiter_name not in self.limiters:
            self.logger.error(f"未知的限流器: {limiter_name}")
            return
        
        limiter = self.limiters[limiter_name]
        
        # 更新配置
        if capacity is not None:
            limiter.config.capacity = capacity
            if isinstance(limiter.impl, TokenBucket):
                limiter.impl.capacity = capacity
            elif isinstance(limiter.impl, SlidingWindowCounter):
                limiter.impl.capacity = capacity
        
        if refill_rate is not None and isinstance(limiter.impl, TokenBucket):
            limiter.config.refill_rate = refill_rate
            limiter.impl.refill_rate = refill_rate
        
        self.logger.info(f"限流配置已更新: {limiter_name}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取限流摘要"""
        summary = {
            'limiters': {},
            'endpoints': {},
            'total_queued': 0
        }
        
        # 限流器摘要
        for name, limiter in self.limiters.items():
            summary['limiters'][name] = {
                'acceptance_rate': limiter.stats.acceptance_rate(),
                'rejected_count': limiter.stats.rejected_requests
            }
        
        # 端点摘要
        for endpoint, stats in self.endpoint_stats.items():
            if stats.total_requests > 0:
                summary['endpoints'][endpoint] = {
                    'total': stats.total_requests,
                    'acceptance_rate': stats.acceptance_rate()
                }
        
        # 总队列大小
        with self.queue_lock:
            summary['total_queued'] = sum(
                len(queue) for queue in self.request_queues.values()
            )
        
        return summary