"""
Script de fine-tuning pour l'agent PPO Pong
Permet de reprendre l'entraînement depuis un modèle existant
avec un système complet de suivi des performances
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Register ALE environments
gym.register_envs(ale_py)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== CNN Policy Network ====================
class CNNPolicy(nn.Module):
    """Convolutional Neural Network for processing Atari frames"""
    def __init__(self, input_channels, n_actions):
        super(CNNPolicy, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc_input_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value
    
    def get_action(self, state):
        """Sample action from policy"""
        with torch.no_grad():
            logits, value = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob.detach(), value.detach()
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy


# ==================== Experience Buffer ====================
class RolloutBuffer:
    """Storage for trajectories collected during rollout"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
    def get(self):
        return (self.states, self.actions, self.rewards, 
                self.values, self.log_probs, self.dones)


# ==================== PPO Agent ====================
class PPOAgent:
    """Proximal Policy Optimization Agent"""
    def __init__(self, input_channels, n_actions, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, c1=0.5, c2=0.01,
                 epochs=2, batch_size=64):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.policy = CNNPolicy(input_channels, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=1.0,  # LR initial
        end_factor=0.1,    # LR final = 0.1 × initial
        total_iters=1000   # nombre total d’updates avant d’atteindre end_factor
    )
        self.buffer = RolloutBuffer()
        
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, value = self.policy.get_action(state_tensor)
        return action, log_prob, value
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(device)
        
        return advantages, returns
    
    def update(self, next_state):
        """Update policy using PPO algorithm"""
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            _, next_value = self.policy(next_state_tensor)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        actions_tensor = torch.LongTensor(actions).to(device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        clip_fractions = []
        
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                log_probs, values, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Calculate clip fraction
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()
                clip_fractions.append(clip_fraction)
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1
        self.scheduler.step()
        self.buffer.clear()
        self.c2 = max(0.001, self.c2 * 0.995)

        return {
            'total_loss': total_loss / update_count,
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count,
            'clip_fraction': np.mean(clip_fractions)
        }


# ==================== Performance Tracker ====================
class PerformanceTracker:
    """Track and save performance metrics during training"""
    def __init__(self, save_dir, tensorboard=True):
        self.save_dir = save_dir
        self.metrics_file = os.path.join(save_dir, "metrics.json")
        self.config_file = os.path.join(save_dir, "config.json")
        
        self.episode_metrics = []
        self.update_metrics = []
        
        # TensorBoard writer
        self.writer = None
        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
        
    def log_episode(self, episode, reward, length, global_step):
        """Log episode metrics"""
        metric = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'global_step': global_step,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_metrics.append(metric)
        
        if self.writer:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Length', length, episode)
            self.writer.add_scalar('Episode/Global_Step', global_step, episode)
    
    def log_update(self, episode, losses):
        """Log update metrics"""
        metric = {
            'episode': episode,
            **losses,
            'timestamp': datetime.now().isoformat()
        }
        self.update_metrics.append(metric)
        
        if self.writer:
            self.writer.add_scalar('Loss/Total', losses['total_loss'], episode)
            self.writer.add_scalar('Loss/Policy', losses['policy_loss'], episode)
            self.writer.add_scalar('Loss/Value', losses['value_loss'], episode)
            self.writer.add_scalar('Loss/Entropy', losses['entropy'], episode)
            if 'clip_fraction' in losses:
                self.writer.add_scalar('PPO/Clip_Fraction', losses['clip_fraction'], episode)
    
    def log_config(self, config):
        """Save training configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def save_metrics(self):
        """Save all metrics to JSON file"""
        metrics = {
            'episodes': self.episode_metrics,
            'updates': self.update_metrics
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def get_statistics(self, window=100):
        """Get recent statistics"""
        if len(self.episode_metrics) < window:
            window = len(self.episode_metrics)
        
        if window == 0:
            return {}
        
        recent_rewards = [m['reward'] for m in self.episode_metrics[-window:]]
        recent_lengths = [m['length'] for m in self.episode_metrics[-window:]]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'total_episodes': len(self.episode_metrics)
        }
    
    def close(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


# ==================== Preprocessing ====================
def preprocess_observation(obs):
    """Preprocess Atari frame"""
    if obs is None:
        return np.zeros((1, 84, 84), dtype=np.float32)
    
    gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
    
    from scipy.ndimage import zoom
    import cv2
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    return resized.astype(np.float32)


def stack_frames(stacked_frames, frame, is_new_episode, stack_size=4):
    """Stack frames for temporal information"""
    frame = preprocess_observation(frame)
    
    if is_new_episode:
        stacked_frames = deque([frame for _ in range(stack_size)], maxlen=stack_size)
    else:
        stacked_frames.append(frame)
    
    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames

def finetune_ppo_pong(base_model_path, n_episodes=500, max_steps=10000, 
                      update_every=2048, save_every=50, log_every=10,
                      lr=5e-5, gamma=0.99, clip_epsilon=0.2):
    """
    Fine-tune PPO agent on Pong with performance tracking
    """
    print("\n" + "="*60)
    print("FINE-TUNING PPO AGENT ON PONG")
    print("="*60)
    
    env = gym.make('ALE/Pong-v5', render_mode=None)
    n_actions = env.action_space.n
    
    print(f"\n📋 Configuration:")
    print(f"  Base model: {base_model_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Learning rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  Clip epsilon: {clip_epsilon}")
    
    agent = PPOAgent(
        input_channels=4,
        n_actions=n_actions,
        lr=lr,
        gamma=gamma,
        gae_lambda=0.95,
        clip_epsilon=clip_epsilon,
        c1=0.3,
        c2=0.01,
        epochs=2,
        batch_size=128
    )
    
    print(f"\n🔄 Loading base model from: {base_model_path}")
    if os.path.exists(base_model_path):
        agent.policy.load_state_dict(torch.load(base_model_path, map_location=device))
        print("✅ Base model loaded successfully!")
    else:
        print("❌ Base model not found! Training from scratch...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"ppo_pong_finetuned_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    tracker = PerformanceTracker(save_dir, tensorboard=True)
    
    config = {
        'base_model': base_model_path,
        'n_episodes': n_episodes,
        'max_steps': max_steps,
        'update_every': update_every,
        'learning_rate': lr,
        'gamma': gamma,
        'clip_epsilon': clip_epsilon,
        'gae_lambda': 0.95,
        'batch_size': 128,
        'epochs': 2,
        'timestamp': timestamp,
        'device': str(device)
    }
    tracker.log_config(config)
    
    print(f"📁 Save directory: {save_dir}")
    print(f"📊 TensorBoard logs: {os.path.join(save_dir, 'tensorboard')}")
    print(f"\n🚀 Starting fine-tuning...\n")
    
    global_step = 0
    best_avg_reward = -float('inf')
    stacked_frames = deque(maxlen=4)
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, observation, True)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            action, log_prob, value = agent.select_action(state)
            agent.buffer.add(state, action, 0, value, log_prob, False)
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state, stacked_frames = stack_frames(stacked_frames, next_observation, False)
            
            agent.buffer.rewards[-1] = reward
            agent.buffer.dones[-1] = done
            
            episode_reward += reward
            state = next_state
            episode_length += 1
            global_step += 1
            
            if global_step % update_every == 0:
                loss_dict = agent.update(state)
                tracker.log_update(episode, loss_dict)
        
        tracker.log_episode(episode, episode_reward, episode_length, global_step)
        
        if (episode + 1) % log_every == 0:
            stats = tracker.get_statistics(window=100)
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes} | Step: {global_step}")
            print(f"{'='*60}")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Avg Reward (last 100): {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
            print(f"  Avg Length (last 100): {stats['avg_length']:.2f}")
            print(f"{'='*60}")
        
        if (episode + 1) % save_every == 0:
            stats = tracker.get_statistics(window=100)
            avg_reward = stats['avg_reward']
            
            checkpoint = {
                'episode': episode,
                'global_step': global_step,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'avg_reward': avg_reward,
                'config': config
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_ep{episode+1}.pth"))
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"\n🏆 New best model saved! Avg reward: {best_avg_reward:.2f}")
            
            tracker.save_metrics()
    
    env.close()
    tracker.close()
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"📁 Results saved to: {save_dir}")
    print(f"📊 View training in TensorBoard:")
    print(f"   tensorboard --logdir={os.path.join(save_dir, 'tensorboard')}")
    print("="*60 + "\n")
    
    return save_dir, tracker


def plot_finetuning_results(save_dir):
    """Plot fine-tuning results from saved metrics"""
    metrics_file = os.path.join(save_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    episodes = metrics['episodes']
    updates = metrics['updates']
    
    episode_nums = [e['episode'] for e in episodes]
    rewards = [e['reward'] for e in episodes]
    lengths = [e['length'] for e in episodes]
    
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    #Episode Rewards
    ax = axes[0, 0]
    ax.plot(episode_nums, rewards, alpha=0.3, color='blue', label='Raw')
    ax.plot(smooth(rewards), alpha=0.8, color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #Episode Lengths
    ax = axes[0, 1]
    ax.plot(episode_nums, lengths, alpha=0.3, color='green', label='Raw')
    ax.plot(smooth(lengths), color='green', alpha=0.8, linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #Rolling Average Reward
    ax = axes[0, 2]
    window = min(100, len(rewards))
    rolling_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    ax.plot(episode_nums, rolling_avg, color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title(f'Rolling Average Reward (window={window})')
    ax.grid(True, alpha=0.3)
    
    #Policy Loss
    ax = axes[1, 0]
    if updates:
        policy_losses = [u['policy_loss'] for u in updates]
        ax.plot(smooth(policy_losses, window=10), alpha=0.8, color='red')
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
    
    #Value Loss
    ax = axes[1, 1]
    if updates:
        value_losses = [u['value_loss'] for u in updates]
        ax.plot(smooth(value_losses, window=10), alpha=0.8, color='orange')
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)
    
    #Entropy
    ax = axes[1, 2]
    if updates:
        entropies = [u['entropy'] for u in updates]
        ax.plot(smooth(entropies, window=10), alpha=0.8, color='cyan')
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetuning_results.png'), dpi=300)
    print(f"\n📊 Results plot saved to: {os.path.join(save_dir, 'finetuning_results.png')}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune PPO agent on Pong")
    parser.add_argument('--model', type=str, default=None,
                       help='Path to base model (default: auto-detect latest)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes for fine-tuning (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--clip', type=float, default=0.2,
                       help='PPO clip epsilon (default: 0.2)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot results after training')
    
    args = parser.parse_args()
    
    if args.model is None:
        import glob
        model_dirs = glob.glob('ppo_pong_*')
        if model_dirs:
            model_dirs.sort(key=os.path.getmtime, reverse=True)
            base_model_path = os.path.join(model_dirs[0], 'best_model.pth')
            print(f"📁 Auto-detected base model: {base_model_path}")
        else:
            print("❌ No base model found! Please train a model first with pong.py")
            exit(1)
    else:
        base_model_path = args.model
    
    save_dir, tracker = finetune_ppo_pong(
        base_model_path=base_model_path,
        n_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip
    )
    
    if args.plot:
        plot_finetuning_results(save_dir)
    
    print("\n💡 To view training progress in TensorBoard, run:")
    print(f"   tensorboard --logdir={os.path.join(save_dir, 'tensorboard')}")
