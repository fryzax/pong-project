#Python script to evaluate a trained model on a number of episodes we choose 

import torch
import gymnasium as gym
import numpy as np
from finetune_pong import PPOAgent, stack_frames

def evaluate(model_path, n_episodes=3, render=True):
    env = gym.make('ALE/Pong-v5', render_mode="human" if render else "rgb_array")
    agent = PPOAgent(input_channels=4, n_actions=env.action_space.n)
    agent.policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.policy.eval()

    print(f"\n🎮 Evaluation du modèle : {model_path}\n")
    all_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        frames = []
        stacked_frames = None
        state, stacked_frames = stack_frames(None, obs, True)

        while not done:
            with torch.no_grad():
                action, _, _ = agent.policy.get_action(torch.FloatTensor(state).unsqueeze(0))
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            state, stacked_frames = stack_frames(stacked_frames, obs, False)
            total_reward += reward


        print(f"Episode {ep+1}: Reward = {total_reward}")
        all_rewards.append(total_reward)

    env.close()
    print(f"\n✅ Moyenne des rewards : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n")


if __name__ == "__main__":
    model_path = "ppo_pong_final_best.pth"
    evaluate(model_path, n_episodes=3, render=True)
