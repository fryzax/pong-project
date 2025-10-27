#Python script for the training of the model from scratch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNNPolicy(nn.Module):
    """
    Convolutional Neural Network for processing Atari frames
    Outputs both policy (actor) and value (critic) estimates
    """
    def __init__(self, input_channels, n_actions):
        super(CNNPolicy, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions (for 210x160 input after preprocessing to 84x84)
        self.fc_input_size = 64 * 7 * 7  # After convolutions on 84x84 image
        
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
            
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class RolloutBuffer:
    """
    Storage for trajectories collected during rollout
    """
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


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """
    def __init__(self, input_channels, n_actions, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, c1=0.5, c2=0.01,
                 epochs=4, batch_size=64):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.policy = CNNPolicy(input_channels, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer()
        
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, value = self.policy.get_action(state_tensor)
        return action, log_prob, value
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
                
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(device)
        
        return advantages, returns
    
    def update(self, next_state):
        """
        Update policy using PPO algorithm
        """
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            _, next_value = self.policy(next_state_tensor)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
        
        # PPO update for multiple epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
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
        
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / update_count,
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count
        }


def preprocess_observation(obs):
    """
    Preprocess Atari frame:
    - Convert to grayscale
    - Resize to 84x84
    - Normalize
    """
    if obs is None:
        return np.zeros((1, 84, 84), dtype=np.float32)
    
    # Convert to grayscale
    gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Resize to 84x84
    from scipy.ndimage import zoom
    resized = zoom(gray, (84/210, 84/160), order=1)
    
    return resized.astype(np.float32)


def stack_frames(stacked_frames, frame, is_new_episode, stack_size=4):
    """
    Stack frames for temporal information
    """
    frame = preprocess_observation(frame)
    
    if is_new_episode:
        stacked_frames = deque([frame for _ in range(stack_size)], maxlen=stack_size)
    else:
        stacked_frames.append(frame)
    
    stacked_state = np.stack(stacked_frames, axis=0)
    
    return stacked_state, stacked_frames


def train_ppo_pong(n_episodes=5000, max_steps=10000, update_every=2048,
                   save_every=100, log_every=10):
    """
    Train PPO agent on single-agent Pong (Gymnasium)
    """
    env = gym.make('ALE/Pong-v5', render_mode=None)
    
    n_actions = env.action_space.n
    
    print(f"Environment: Pong (Gymnasium)")
    print(f"Number of actions: {n_actions}")
    print(f"Observation space: {env.observation_space.shape}")
    
    agent = PPOAgent(
        input_channels=4,  # Stacked frames
        n_actions=n_actions,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,
        c2=0.01,
        epochs=4,
        batch_size=256
    )
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    losses = {'policy': [], 'value': [], 'entropy': []}
    
    stacked_frames = deque(maxlen=4)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"ppo_pong_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    global_step = 0
    best_avg_reward = -float('inf')
    
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
                losses['policy'].append(loss_dict['policy_loss'])
                losses['value'].append(loss_dict['value_loss'])
                losses['entropy'].append(loss_dict['entropy'])
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % log_every == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes} | Global Step: {global_step}")
            print(f"Avg Episode Length (last 100): {avg_length:.2f}")
            print(f"Avg Reward (last 100): {avg_reward:.2f}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"{'='*60}")
        
        if (episode + 1) % save_every == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(
                    agent.policy.state_dict(),
                    os.path.join(save_dir, "best_model.pth")
                )
                print(f"\n🏆 New best model saved! Avg reward: {best_avg_reward:.2f}")
            
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': episode_rewards,
            }, os.path.join(save_dir, f"checkpoint_ep{episode+1}.pth"))
    
    env.close()
    
    return episode_rewards, episode_lengths, losses, save_dir


def plot_training_curves(episode_rewards, episode_lengths, losses, save_dir):
    """
    Plot and save training curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    #Episode Rewards
    ax = axes[0, 0]
    rewards = episode_rewards
    ax.plot(smooth(rewards), alpha=0.8, color='blue')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    #Episode Lengths
    ax = axes[0, 1]
    ax.plot(smooth(episode_lengths), color='green', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Lengths (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    #Policy Loss
    ax = axes[1, 0]
    if losses['policy']:
        ax.plot(smooth(losses['policy'], window=10), alpha=0.8, color='red')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)
    
    #Value Loss
    ax = axes[1, 1]
    if losses['value']:
        ax.plot(smooth(losses['value'], window=10), alpha=0.8, color='orange')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    print(f"\n📊 Training curves saved to {save_dir}/training_curves.png")
    plt.show()


def evaluate_agents(agent, n_episodes=10, render=True):
    """
    Evaluate trained agent
    """
    render_mode = "human" if render else None
    env = gym.make('ALE/Pong-v5', render_mode=render_mode)
    
    total_rewards = []
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        
        stacked_frames = deque(maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, observation, True)
        
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = agent.policy(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state, stacked_frames = stack_frames(stacked_frames, next_observation, False)
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"{'='*60}\n")
    
    return total_rewards


if __name__ == "__main__":
    print("="*60)
    print("PPO FOR ATARI PONG (GYMNASIUM)")
    print("="*60)
    
    print("\n🚀 Starting training...")
    episode_rewards, episode_lengths, losses, save_dir = train_ppo_pong(
        n_episodes=1000,      
        max_steps=10000,      
        update_every=2048,    
        save_every=50,        
        log_every=10          
    )
    
    print("\n📊 Generating training curves...")
    plot_training_curves(episode_rewards, episode_lengths, losses, save_dir)
    
    print("\n🎮 Evaluating trained agent...")
    
    env = gym.make('ALE/Pong-v5', render_mode=None)
    n_actions = env.action_space.n
    env.close()
    
    agent = PPOAgent(input_channels=4, n_actions=n_actions)
    agent.policy.load_state_dict(
        torch.load(os.path.join(save_dir, "best_model.pth"))
    )
    agent.policy.eval()
    
    evaluate_agents(agent, n_episodes=20, render=False)
    
    print("\n✅ Training and evaluation complete!")
    print(f"📁 All files saved to: {save_dir}")
