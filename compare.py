#Python Script to compare two trained models to compare the fine-tuning performance

import torch
import gymnasium as gym
import numpy as np
from finetune_pong import PPOAgent, stack_frames

def evaluate_model(model_path, n_episodes=20):
    env = gym.make('ALE/Pong-v5', render_mode=None)
    agent = PPOAgent(input_channels=4, n_actions=env.action_space.n)
    agent.policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.policy.eval()

    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        frames = None
        state, frames = stack_frames(None, obs, True)
        
        while not done:
            with torch.no_grad():
                action, _, _ = agent.policy.get_action(torch.FloatTensor(state).unsqueeze(0))
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            state, frames = stack_frames(frames, obs, False)
            total_reward += reward
        rewards.append(total_reward)

    env.close()
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    base_model = "ppo_pong_finetuned_20251024_194634/best_model.pth"     
    finetuned_model = "ppo_pong_final_best.pth"                 
    base_mean, base_std = evaluate_model(base_model)
    fine_mean, fine_std = evaluate_model(finetuned_model)

    print("\nModel Comparison")
    print("──────────────────────────")
    print(f"Before fine-tuning  : {base_mean:.2f} ± {base_std:.2f}")
    print(f"After fine-tuning  : {fine_mean:.2f} ± {fine_std:.2f}")
    print(f"Average gain         : {fine_mean - base_mean:.2f} rewards points\n")
