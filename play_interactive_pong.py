"""
Script pour jouer INTERACTIVEMENT contre l'agent PPO entraîné
Contrôles clavier :
- Flèche HAUT : Monter votre raquette
- Flèche BAS : Descendre votre raquette
- ESC ou Q : Quitter
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
import pygame
from pygame.locals import *

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


# ==================== Interactive Play Function ====================
def play_interactive_against_agent(model_path, n_games=5):
    """
    Jouer INTERACTIVEMENT contre l'agent entraîné
    Le joueur contrôle sa raquette avec les flèches du clavier
    """
    # Initialize pygame
    pygame.init()
    
    # Create environment with rgb_array for pygame display
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    n_actions = env.action_space.n
    
    # Pong action mapping:
    # 0 = NOOP, 1 = FIRE, 2 = RIGHT (UP), 3 = LEFT (DOWN), 4 = RIGHTFIRE, 5 = LEFTFIRE
    
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
        pygame.quit()
        return
    
    # Setup pygame window
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pong - Vous vs Agent PPO")
    clock = pygame.time.Clock()
    
    # Font for text
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    print("\n" + "="*60)
    print("🎮 MODE JEU INTERACTIF - Vous jouez contre l'agent PPO!")
    print("="*60)
    print("\n📋 CONTRÔLES:")
    print("  - Flèche HAUT : Monter votre raquette")
    print("  - Flèche BAS : Descendre votre raquette")
    print("  - ESC ou Q : Quitter")
    print("\n⚠️  Note: Votre raquette est à DROITE, l'agent est à GAUCHE")
    print("="*60 + "\n")
    
    wins = 0
    losses = 0
    
    for game in range(n_games):
        observation, info = env.reset()
        
        # Initialize frame stacking
        stacked_frames = deque(maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, observation, True)
        
        game_reward = 0
        player_score = 0
        agent_score = 0
        done = False
        frame_count = 0
        
        # Player action (0 = NOOP by default)
        player_action = 0
        
        print(f"\n🎯 Partie {game + 1}/{n_games}")
        print("Appuyez sur les flèches HAUT/BAS pour jouer!")
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    env.close()
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        pygame.quit()
                        env.close()
                        return
            
            # Get keyboard input for player
            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                player_action = 2  # RIGHT (UP)
            elif keys[K_DOWN]:
                player_action = 3  # LEFT (DOWN)
            else:
                player_action = 0  # NOOP
            
            # Agent selects action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = agent_policy(state_tensor)
                probs = F.softmax(logits, dim=-1)
                agent_action = torch.argmax(probs, dim=-1).item()
            
            # Use player action instead of agent action
            action = player_action
            
            # Step environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update scores based on reward
            if reward > 0:
                player_score += 1
            elif reward < 0:
                agent_score += 1
            
            # Update state
            next_state, stacked_frames = stack_frames(stacked_frames, next_observation, False)
            state = next_state
            game_reward += reward
            frame_count += 1
            
            # Render game
            frame = env.render()
            if frame is not None:
                # Convert frame to pygame surface
                frame = np.transpose(frame, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                surf = pygame.transform.scale(surf, (screen_width, screen_height))
                screen.blit(surf, (0, 0))
                
                # Draw scores
                score_text = font.render(f"Agent: {agent_score}  Vous: {player_score}", True, (255, 255, 255))
                screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, 10))
                
                # Draw controls reminder
                controls_text = small_font.render("↑/↓ pour bouger | ESC pour quitter", True, (200, 200, 200))
                screen.blit(controls_text, (screen_width // 2 - controls_text.get_width() // 2, screen_height - 30))
                
                pygame.display.flip()
                clock.tick(15)  # 15 FPS (ralenti pour mieux jouer)
            
            # Print score updates
            if reward != 0:
                if reward > 0:
                    print(f"  🏆 Vous marquez! Score: Agent {agent_score} - Vous {player_score}")
                else:
                    print(f"  😞 Agent marque! Score: Agent {agent_score} - Vous {player_score}")
        
        # Game result
        print(f"\n📊 Fin de la partie {game + 1}")
        print(f"  Score final: Agent {agent_score} - Vous {player_score}")
        print(f"  Nombre de frames: {frame_count}")
        
        if player_score > agent_score:
            wins += 1
            print("  ✅ Vous gagnez cette partie!")
        elif agent_score > player_score:
            losses += 1
            print("  ❌ L'agent gagne cette partie!")
        else:
            print("  🤝 Égalité!")
        
        # Wait a bit before next game
        pygame.time.wait(2000)
    
    env.close()
    pygame.quit()
    
    # Final statistics
    print("\n" + "="*60)
    print("📈 STATISTIQUES FINALES")
    print("="*60)
    print(f"Parties jouées: {n_games}")
    print(f"Vos victoires: {wins} ({wins/n_games*100:.1f}%)")
    print(f"Victoires de l'agent: {losses} ({losses/n_games*100:.1f}%)")
    print(f"Égalités: {n_games - wins - losses}")
    print("="*60 + "\n")


# ==================== Main ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jouer INTERACTIVEMENT contre l'agent PPO")
    parser.add_argument('--model', type=str, default=None,
                       help='Chemin vers le modèle entraîné (par défaut: cherche dans ppo_pong_*/best_model.pth)')
    parser.add_argument('--games', type=int, default=3,
                       help='Nombre de parties à jouer (défaut: 3)')
    
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
        play_interactive_against_agent(model_path, n_games=args.games)
    except KeyboardInterrupt:
        print("\n\n👋 Partie interrompue. À bientôt!")
        pygame.quit()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
