import torch
from finetune_pong import PPOAgent
import os

# === Charger ton dernier modèle fine-tuné ===
model_path = "ppo_pong_finetuned_20251024_213901/best_model.pth"  # ← remplace par ton dossier exact
agent = PPOAgent(input_channels=4, n_actions=6)  # Pong a 6 actions
agent.policy.load_state_dict(torch.load(model_path, map_location='cpu'))

# === Sauvegarder officiellement la version finale ===
save_path = "ppo_pong_final_best.pth"
torch.save(agent.policy.state_dict(), save_path)

print(f"✅ Modèle final sauvegardé sous : {save_path}")
