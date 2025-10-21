"""
Script pour jouer contre l'agent PPO entraîné
Contrôles clavier :
- Flèche HAUT : Monter la raquette
- Flèche BAS : Descendre la raquette
- Q : Quitter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import sys
import os

# Register ALE environments
gym.register_envs(ale_py)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CNN Policy Network ====================
class CNNPolicy(nn.Module):
    """
    Convolutional Neural Network for processing Atari frames
    """
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


# ==================== Preprocessing ====================
def preprocess_observation(obs):
    """Preprocess Atari frame"""
    if obs is None:
        return np.zeros((1, 84, 84), dtype=np.float32)
    
    # Convert to grayscale
    gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Resize to 84x84
    from scipy.ndimage import zoom
    resized = zoom(gray, (84/210, 84/160), order=1)
    
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


# ==================== Play Function ====================
def play_against_agent(model_path, n_games=5):
    """
    Jouer contre l'agent entraîné
    """
    # Create environment with human rendering
    env = gym.make('ALE/Pong-v5', render_mode='human')
    n_actions = env.action_space.n
    
    # Load trained agent
    print("🤖 Chargement de l'agent entraîné...")
    agent_policy = CNNPolicy(input_channels=4, n_actions=n_actions).to(device)
    
    if os.path.exists(model_path):
        agent_policy.load_state_dict(torch.load(model_path, map_location=device))
        agent_policy.eval()
        print(f"✅ Modèle chargé depuis: {model_path}")
    else:
        print(f"❌ Erreur: Le modèle {model_path} n'existe pas!")
        print("Veuillez d'abord entraîner l'agent avec pong.py")
        return
    
    print("\n" + "="*60)
    print("🎮 MODE JEU - Vous jouez contre l'agent PPO!")
    print("="*60)
    print("\n📋 CONTRÔLES:")
    print("  - Votre raquette est contrôlée automatiquement")
    print("  - L'agent contrôle l'autre raquette")
    print("  - Appuyez sur 'Q' dans le terminal pour quitter")
    print("\n⚠️  Note: Pong Atari nécessite ~100 frames pour démarrer")
    print("="*60 + "\n")
    
    wins = 0
    losses = 0
    
    for game in range(n_games):
        observation, info = env.reset()
        
        # Initialize frame stacking
        stacked_frames = deque(maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, observation, True)
        
        game_reward = 0
        done = False
        frame_count = 0
        
        print(f"\n🎯 Partie {game + 1}/{n_games}")
        
        while not done:
            # Agent selects action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = agent_policy(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            # Step environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state
            next_state, stacked_frames = stack_frames(stacked_frames, next_observation, False)
            state = next_state
            game_reward += reward
            frame_count += 1
            
            # Print score updates
            if reward != 0:
                if reward > 0:
                    print(f"  🏆 Agent marque! (+{reward:.0f})")
                else:
                    print(f"  😞 Vous marquez! ({reward:.0f})")
        
        # Game result
        print(f"\n📊 Fin de la partie {game + 1}")
        print(f"  Score final: {game_reward:.0f}")
        print(f"  Nombre de frames: {frame_count}")
        
        if game_reward > 0:
            wins += 1
            print("  ✅ L'agent gagne cette partie!")
        elif game_reward < 0:
            losses += 1
            print("  ❌ Vous gagnez cette partie!")
        else:
            print("  🤝 Égalité!")
    
    env.close()
    
    # Final statistics
    print("\n" + "="*60)
    print("📈 STATISTIQUES FINALES")
    print("="*60)
    print(f"Parties jouées: {n_games}")
    print(f"Victoires de l'agent: {wins} ({wins/n_games*100:.1f}%)")
    print(f"Vos victoires: {losses} ({losses/n_games*100:.1f}%)")
    print(f"Égalités: {n_games - wins - losses}")
    print("="*60 + "\n")


# ==================== Demo Mode ====================
def watch_agent_play(model_path, n_games=3):
    """
    Regarder l'agent jouer seul
    """
    # Create environment with human rendering
    env = gym.make('ALE/Pong-v5', render_mode='human')
    n_actions = env.action_space.n
    
    # Load trained agent
    print("🤖 Chargement de l'agent entraîné...")
    agent_policy = CNNPolicy(input_channels=4, n_actions=n_actions).to(device)
    
    if os.path.exists(model_path):
        agent_policy.load_state_dict(torch.load(model_path, map_location=device))
        agent_policy.eval()
        print(f"✅ Modèle chargé depuis: {model_path}")
    else:
        print(f"❌ Erreur: Le modèle {model_path} n'existe pas!")
        return
    
    print("\n" + "="*60)
    print("👁️  MODE DÉMO - Regardez l'agent jouer!")
    print("="*60 + "\n")
    
    total_reward = 0
    
    for game in range(n_games):
        observation, info = env.reset()
        
        stacked_frames = deque(maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, observation, True)
        
        game_reward = 0
        done = False
        
        print(f"🎮 Partie {game + 1}/{n_games}")
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = agent_policy(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state, stacked_frames = stack_frames(stacked_frames, next_observation, False)
            state = next_state
            game_reward += reward
            
            if reward != 0:
                if reward > 0:
                    print(f"  ✅ Point marqué! (+{reward:.0f})")
                else:
                    print(f"  ❌ Point perdu! ({reward:.0f})")
        
        total_reward += game_reward
        print(f"  Score de la partie: {game_reward:.0f}\n")
    
    env.close()
    
    print("="*60)
    print(f"📊 Score total: {total_reward:.0f}")
    print(f"📈 Score moyen: {total_reward/n_games:.2f}")
    print("="*60 + "\n")


# ==================== Main ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jouer contre l'agent PPO entraîné")
    parser.add_argument('--model', type=str, default=None,
                       help='Chemin vers le modèle entraîné (par défaut: cherche dans ppo_pong_*/best_model.pth)')
    parser.add_argument('--games', type=int, default=3,
                       help='Nombre de parties à jouer (défaut: 3)')
    parser.add_argument('--demo', action='store_true',
                       help='Mode démo: regarder l\'agent jouer seul')
    
    args = parser.parse_args()
    
    # Find the most recent trained model if not specified
    if args.model is None:
        import glob
        model_dirs = glob.glob('ppo_pong_*')
        if model_dirs:
            # Sort by modification time
            model_dirs.sort(key=os.path.getmtime, reverse=True)
            model_path = os.path.join(model_dirs[0], 'best_model.pth')
            print(f"📁 Utilisation du modèle: {model_path}")
        else:
            print("❌ Aucun modèle trouvé!")
            print("Veuillez d'abord entraîner un agent avec: python pong.py")
            sys.exit(1)
    else:
        model_path = args.model
    
    try:
        if args.demo:
            watch_agent_play(model_path, n_games=args.games)
        else:
            play_against_agent(model_path, n_games=args.games)
    except KeyboardInterrupt:
        print("\n\n👋 Partie interrompue. À bientôt!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
