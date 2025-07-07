# train_rl.py - RL模型训练
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import json
from collections import deque

from logger import TradingLogger
from utils import ConfigManager, TimeUtils
from rl_environment import TradingEnvironment
from prepare_rl_data import TrajectoryBuffer, StateSpace
from database_manager import DatabaseManager


class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Actor头（策略网络）
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic头（价值网络）
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        shared_features = self.shared(state)
        action_probs = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        return action_probs, state_value


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("PPO_AGENT")
        
        # 网络参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = self.config.get("rl.ppo.hidden_dim", 256)
        
        # PPO超参数
        self.lr = self.config.get("rl.ppo.learning_rate", 3e-4)
        self.gamma = self.config.get("rl.ppo.gamma", 0.99)
        self.gae_lambda = self.config.get("rl.ppo.gae_lambda", 0.95)
        self.clip_epsilon = self.config.get("rl.ppo.clip_epsilon", 0.2)
        self.entropy_coef = self.config.get("rl.ppo.entropy_coef", 0.01)
        self.value_loss_coef = self.config.get("rl.ppo.value_loss_coef", 0.5)
        self.max_grad_norm = self.config.get("rl.ppo.max_grad_norm", 0.5)
        
        # 训练参数
        self.n_epochs = self.config.get("rl.ppo.n_epochs", 10)
        self.batch_size = self.config.get("rl.ppo.batch_size", 64)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.policy_net = ActorCriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 训练统计
        self.training_stats = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'entropy_losses': deque(maxlen=100),
            'rewards': deque(maxlen=100)
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, state_value = self.policy_net(state_tensor)
            
            if deterministic:
                action = action_probs.argmax().item()
                log_prob = torch.log(action_probs[0, action])
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()
            
            return action, log_prob.item(), state_value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool]) -> Tuple[List[float], List[float]]:
        """计算广义优势估计(GAE)"""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_value = values[t]
        
        return advantages, returns
    
    def update(self, trajectories: Dict[str, List]) -> Dict[str, float]:
        """更新网络"""
        # 转换为张量
        states = torch.FloatTensor(trajectories['states']).to(self.device)
        actions = torch.LongTensor(trajectories['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectories['log_probs']).to(self.device)
        advantages = torch.FloatTensor(trajectories['advantages']).to(self.device)
        returns = torch.FloatTensor(trajectories['returns']).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练多个epoch
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            perm = torch.randperm(states.size(0))
            
            for i in range(0, states.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                
                # 获取批次数据
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 前向传播
                action_probs, state_values = self.policy_net(batch_states)
                dist = Categorical(action_probs)
                
                # 计算新的log概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(state_values.squeeze(), batch_returns)
                
                # 熵损失（鼓励探索）
                entropy_loss = -dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # 更新统计
        num_updates = (states.size(0) // self.batch_size) * self.n_epochs
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        self.training_stats['entropy_losses'].append(avg_entropy_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss
        }
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': dict(self.training_stats),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim
            }
        }, filepath)
        self.logger.info(f"PPO模型已保存: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', dict(self.training_stats))
        self.logger.info(f"PPO模型已加载: {filepath}")


class A2CAgent:
    """A2C智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("A2C_AGENT")
        
        # 网络参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = self.config.get("rl.a2c.hidden_dim", 256)
        
        # A2C超参数
        self.lr = self.config.get("rl.a2c.learning_rate", 3e-4)
        self.gamma = self.config.get("rl.a2c.gamma", 0.99)
        self.entropy_coef = self.config.get("rl.a2c.entropy_coef", 0.01)
        self.value_loss_coef = self.config.get("rl.a2c.value_loss_coef", 0.5)
        self.max_grad_norm = self.config.get("rl.a2c.max_grad_norm", 0.5)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.policy_net = ActorCriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 训练统计
        self.training_stats = {
            'losses': deque(maxlen=100),
            'rewards': deque(maxlen=100)
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, state_value = self.policy_net(state_tensor)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), state_value.item()
    
    def update(self, transitions: List[Dict]) -> float:
        """更新网络（在线更新）"""
        # 准备数据
        states = torch.FloatTensor([t['state'] for t in transitions]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in transitions]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in transitions]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in transitions]).to(self.device)
        
        # 前向传播
        action_probs, state_values = self.policy_net(states)
        _, next_state_values = self.policy_net(next_states)
        
        # 计算TD目标
        td_targets = rewards + self.gamma * next_state_values.squeeze() * (1 - dones)
        
        # 计算优势
        advantages = td_targets - state_values.squeeze()
        
        # 计算损失
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(state_values.squeeze(), td_targets.detach())
        entropy_loss = -dist.entropy().mean()
        
        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 记录统计
        self.training_stats['losses'].append(total_loss.item())
        
        return total_loss.item()


class RLTrainer:
    """RL训练器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("RL_TRAINER")
        
        # 训练配置
        self.algorithm = self.config.get("rl.algorithm", "ppo")  # ppo or a2c
        self.num_episodes = self.config.get("rl.training.num_episodes", 1000)
        self.save_interval = self.config.get("rl.training.save_interval", 100)
        self.eval_interval = self.config.get("rl.training.eval_interval", 50)
        self.early_stopping_patience = self.config.get("rl.training.early_stopping_patience", 20)
        
        # 经验回放
        self.use_experience_replay = self.config.get("rl.use_experience_replay", True)
        self.replay_buffer_size = self.config.get("rl.replay_buffer_size", 100000)
        
        # 初始化组件
        self.env = TradingEnvironment(config_path)
        self.state_space = StateSpace(config_path)
        self.trajectory_buffer = TrajectoryBuffer(self.replay_buffer_size)
        self.db_manager = DatabaseManager(config_path)
        
        # 模型保存路径
        self.model_dir = Path("models/rl")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.training_history = []
        self.best_reward = -float('inf')
        self.patience_counter = 0
    
    def train(self, symbol: str, market_data: pd.DataFrame, 
             selected_features: List[str] = None):
        """训练RL模型"""
        self.logger.info(f"开始训练 {self.algorithm.upper()} 模型 - {symbol}")
        
        # 设置环境数据
        self.env.set_market_data(market_data, selected_features)
        
        # 创建智能体
        state_dim = self.state_space.total_dim
        action_dim = self.env.action_space.n
        
        if self.algorithm == 'ppo':
            agent = PPOAgent(state_dim, action_dim, self.config.config_path)
        else:
            agent = A2CAgent(state_dim, action_dim, self.config.config_path)
        
        # 训练循环
        for episode in range(self.num_episodes):
            episode_reward, episode_length = self._train_episode(agent, episode)
            
            # 记录训练历史
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'timestamp': TimeUtils.now_timestamp()
            })
            
            # 评估
            if episode % self.eval_interval == 0:
                eval_reward = self._evaluate(agent, num_episodes=5)
                self.logger.info(f"Episode {episode} - Train Reward: {episode_reward:.2f}, "
                               f"Eval Reward: {eval_reward:.2f}")
                
                # 早停检查
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.patience_counter = 0
                    # 保存最佳模型
                    best_model_path = self.model_dir / f"{symbol}_{self.algorithm}_best.pth"
                    agent.save(str(best_model_path))
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info("早停触发，停止训练")
                    break
            
            # 定期保存
            if episode % self.save_interval == 0:
                model_path = self.model_dir / f"{symbol}_{self.algorithm}_episode_{episode}.pth"
                agent.save(str(model_path))
        
        # 保存训练历史
        self._save_training_history(symbol)
        
        return agent
    
    def _train_episode(self, agent, episode: int) -> Tuple[float, int]:
        """训练单个回合"""
        state = self.env.reset()
        
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        episode_reward = 0
        step = 0
        
        while True:
            # 选择动作
            if isinstance(agent, PPOAgent):
                action, log_prob, value = agent.select_action(state)
            else:
                action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录轨迹
            trajectories['states'].append(state)
            trajectories['actions'].append(action)
            trajectories['rewards'].append(reward)
            trajectories['log_probs'].append(log_prob)
            trajectories['values'].append(value)
            trajectories['dones'].append(done)
            
            # 经验回放
            if self.use_experience_replay:
                self.trajectory_buffer.add(state, action, reward, next_state, done, info)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        # PPO需要批量更新
        if isinstance(agent, PPOAgent):
            # 计算优势和回报
            advantages, returns = agent.compute_gae(
                trajectories['rewards'],
                trajectories['values'],
                trajectories['dones']
            )
            
            trajectories['advantages'] = advantages
            trajectories['returns'] = returns
            
            # 更新网络
            update_info = agent.update(trajectories)
            
        # A2C在线更新
        else:
            # 准备转换数据
            transitions = []
            for i in range(len(trajectories['states'])-1):
                transitions.append({
                    'state': trajectories['states'][i],
                    'action': trajectories['actions'][i],
                    'reward': trajectories['rewards'][i],
                    'next_state': trajectories['states'][i+1],
                    'done': trajectories['dones'][i]
                })
            
            if transitions:
                agent.update(transitions)
        
        return episode_reward, step
    
    def _evaluate(self, agent, num_episodes: int = 5) -> float:
        """评估智能体"""
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                # 确定性动作选择
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.select_action(state, deterministic=True)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        action_probs, _ = agent.policy_net(state_tensor)
                        action = action_probs.argmax().item()
                
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def _save_training_history(self, symbol: str):
        """保存训练历史"""
        history_path = self.model_dir / f"{symbol}_{self.algorithm}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"训练历史已保存: {history_path}")