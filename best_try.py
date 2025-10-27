#Python script to save the best model after fine-tuning

import torch
from finetune_pong import PPOAgent
import os

model_path = "ppo_pong_finetuned_20251024_213901/best_model.pth"  
agent = PPOAgent(input_channels=4, n_actions=6)  
agent.policy.load_state_dict(torch.load(model_path, map_location='cpu'))

save_path = "ppo_pong_final_best.pth"
torch.save(agent.policy.state_dict(), save_path)

print(f"✅ Modèle final sauvegardé sous : {save_path}")
