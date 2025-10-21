# 🏓 Projet PPO Pong - Apprentissage par Renforcement

Ce projet implémente un agent PPO (Proximal Policy Optimization) pour jouer au jeu Atari Pong, avec des outils complets de fine-tuning et d'analyse des performances.

## 📋 Structure du Projet

```
.
├── pong.py                     # Script d'entraînement principal
├── finetune_pong.py           # Script de fine-tuning avec tracking
├── play_pong.py               # Jouer contre l'agent (mode auto)
├── play_interactive_pong.py   # Jouer contre l'agent (mode interactif)
├── analyze_performance.py     # Analyser et comparer les performances
├── ppo_pong_*/                # Dossiers des modèles entraînés
│   ├── best_model.pth        # Meilleur modèle
│   ├── checkpoint_*.pth      # Checkpoints réguliers
│   ├── metrics.json          # Métriques d'entraînement
│   ├── config.json           # Configuration utilisée
│   └── tensorboard/          # Logs TensorBoard
└── README.md                  # Ce fichier
```

## 🚀 Installation

```bash
# Installer les dépendances
pip install torch gymnasium ale-py scipy pygame tensorboard

# Cloner le repo
git clone https://github.com/fryzax/pong-project.git
cd pong-project
```

## 🎯 Utilisation

### 1. Entraînement Initial

```bash
# Entraîner un nouvel agent
python pong.py
```

Paramètres configurables dans le code :
- `n_episodes`: Nombre d'épisodes (défaut: 1000)
- `learning_rate`: Taux d'apprentissage (défaut: 2.5e-4)
- `gamma`: Facteur de discount (défaut: 0.99)
- `update_every`: Fréquence de mise à jour (défaut: 2048)

### 2. Fine-tuning

```bash
# Fine-tuner depuis le meilleur modèle (auto-détection)
python finetune_pong.py --episodes 500

# Fine-tuner depuis un modèle spécifique
python finetune_pong.py --model ppo_pong_20251021_110351/best_model.pth --episodes 500

# Paramètres personnalisés
python finetune_pong.py --episodes 500 --lr 1e-4 --gamma 0.99 --clip 0.2 --plot
```

**Options disponibles:**
- `--model`: Chemin vers le modèle de base (auto-détecte le plus récent si non spécifié)
- `--episodes`: Nombre d'épisodes de fine-tuning (défaut: 500)
- `--lr`: Learning rate (défaut: 1e-4)
- `--gamma`: Facteur de discount (défaut: 0.99)
- `--clip`: Epsilon de clipping PPO (défaut: 0.2)
- `--plot`: Générer les graphiques à la fin

**Suivi avec TensorBoard:**
```bash
# Lancer TensorBoard pour voir l'entraînement en temps réel
tensorboard --logdir=ppo_pong_finetuned_XXXXXXXX_XXXXXX/tensorboard
```

### 3. Jouer contre l'Agent

#### Mode Automatique (regarder jouer)
```bash
python play_pong.py --games 3
```

#### Mode Interactif (vous jouez!)
```bash
python play_interactive_pong.py --games 3
```

**Contrôles:**
- ⬆️ Flèche HAUT : Monter la raquette
- ⬇️ Flèche BAS : Descendre la raquette
- ESC ou Q : Quitter

### 4. Analyser les Performances

```bash
# Comparer tous les modèles entraînés
python analyze_performance.py --plot --report

# Comparer des modèles spécifiques
python analyze_performance.py --models ppo_pong_A ppo_pong_B --plot

# Seulement générer le rapport texte
python analyze_performance.py --report

# Seulement générer les graphiques
python analyze_performance.py --plot --output comparison.png
```

**Sorties:**
- Tableau comparatif dans le terminal
- Graphiques de comparaison (si `--plot`)
- Rapport détaillé en texte (si `--report`)

## 📊 Suivi des Performances

### Métriques Trackées

Le système de fine-tuning enregistre automatiquement :

**Par épisode:**
- Récompense totale
- Longueur de l'épisode
- Step global
- Timestamp

**Par mise à jour:**
- Loss totale
- Policy loss
- Value loss
- Entropie
- Clip fraction (taux de clipping PPO)

### Fichiers Générés

Chaque entraînement crée un dossier `ppo_pong_[finetuned_]YYYYMMDD_HHMMSS/` contenant :

1. **metrics.json**: Toutes les métriques au format JSON
   ```json
   {
     "episodes": [
       {"episode": 0, "reward": -21.0, "length": 1234, ...},
       ...
     ],
     "updates": [
       {"episode": 10, "total_loss": 0.123, ...},
       ...
     ]
   }
   ```

2. **config.json**: Configuration d'entraînement
   ```json
   {
     "learning_rate": 0.0001,
     "gamma": 0.99,
     "clip_epsilon": 0.2,
     "base_model": "ppo_pong_*/best_model.pth",
     ...
   }
   ```

3. **tensorboard/**: Logs TensorBoard pour visualisation interactive

4. **best_model.pth**: Meilleur modèle (moyenne des 100 derniers épisodes)

5. **checkpoint_epXXX.pth**: Checkpoints réguliers avec état complet

## 📈 Visualisation avec TensorBoard

TensorBoard offre une visualisation en temps réel :

```bash
tensorboard --logdir=ppo_pong_finetuned_XXXXXXXX_XXXXXX/tensorboard
```

Puis ouvrir http://localhost:6006 dans votre navigateur.

**Métriques disponibles:**
- Episode/Reward
- Episode/Length
- Episode/Global_Step
- Loss/Total
- Loss/Policy
- Loss/Value
- Loss/Entropy
- PPO/Clip_Fraction

## 🎓 Architecture du Réseau

**CNN Policy Network:**
- Conv1: 32 filtres (8x8, stride=4) + ReLU
- Conv2: 64 filtres (4x4, stride=2) + ReLU
- Conv3: 64 filtres (3x3, stride=1) + ReLU
- FC1: 512 neurones + ReLU
- Actor head: Softmax sur actions
- Critic head: Estimation de valeur

**Preprocessing:**
- Conversion en niveaux de gris
- Resize 84x84
- Stack de 4 frames pour information temporelle

## ⚙️ Hyperparamètres PPO

### Entraînement Initial
- Learning rate: 2.5e-4
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip epsilon: 0.2
- Epochs: 4
- Batch size: 256

### Fine-tuning (Recommandé)
- Learning rate: 1e-4 (plus faible)
- Gamma: 0.99
- Clip epsilon: 0.2
- Batch size: 256

## 📊 Résultats Attendus

**Après entraînement initial (~1000 épisodes):**
- Récompense moyenne: -5 à +5
- Performance stable avec quelques victoires

**Après fine-tuning (~500 épisodes supplémentaires):**
- Récompense moyenne: +5 à +15
- Amélioration de la consistance
- Meilleure anticipation

## 🛠️ Troubleshooting

### Erreur "No module named X"
```bash
pip install torch gymnasium ale-py scipy pygame tensorboard
```

### Fenêtre de jeu ne s'affiche pas
- Vérifier que pygame est installé
- Essayer `play_pong.py` (mode demo) au lieu d'interactif

### Performance faible après fine-tuning
- Réduire le learning rate: `--lr 5e-5`
- Augmenter le nombre d'épisodes: `--episodes 1000`
- Vérifier que le bon modèle de base est chargé

### TensorBoard ne démarre pas
```bash
pip install --upgrade tensorboard
tensorboard --logdir=<path> --host=localhost
```

## 📝 Exemple de Workflow Complet

```bash
# 1. Entraînement initial
python pong.py

# 2. Vérifier les performances
python play_pong.py --games 5

# 3. Fine-tuner pour améliorer
python finetune_pong.py --episodes 500 --lr 1e-4 --plot

# 4. Suivre en temps réel
tensorboard --logdir=ppo_pong_finetuned_*/tensorboard

# 5. Comparer les modèles
python analyze_performance.py --plot --report

# 6. Jouer contre le meilleur modèle
python play_interactive_pong.py --games 3
```

## 🎯 Objectifs de Performance

| Métrique | Initial | Après Fine-tuning | Expert |
|----------|---------|-------------------|--------|
| Récompense moyenne | -10 à 0 | +5 à +10 | +15 à +21 |
| Taux de victoire | 20-40% | 60-80% | 90%+ |
| Longueur moyenne | 1000-2000 | 800-1500 | 500-1000 |

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche (`git checkout -b feature/amélioration`)
3. Commit les changements (`git commit -am 'Ajout fonctionnalité'`)
4. Push la branche (`git push origin feature/amélioration`)
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - Voir LICENSE pour plus de détails

## 👥 Auteurs

- [@fryzax](https://github.com/fryzax)

## 🙏 Remerciements

- OpenAI Gym/Gymnasium pour l'environnement
- Stable Baselines3 pour l'inspiration
- PyTorch pour le framework deep learning
