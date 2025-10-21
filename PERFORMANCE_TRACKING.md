# 📊 Système de Suivi des Performances - Guide Complet

## Vue d'ensemble

Le système de suivi des performances pour le projet PPO Pong offre un tracking complet et automatique de toutes les métriques d'entraînement, avec plusieurs outils d'analyse et de visualisation.

## 🎯 Fonctionnalités Principales

### 1. Tracking Automatique

**Métriques par Épisode:**
- ✅ Récompense totale de l'épisode
- ✅ Longueur de l'épisode (nombre de frames)
- ✅ Step global (compteur cumulatif)
- ✅ Timestamp précis

**Métriques par Mise à Jour:**
- ✅ Loss totale (policy + value + entropy)
- ✅ Policy loss (loss de la politique)
- ✅ Value loss (loss du critique)
- ✅ Entropie (exploration)
- ✅ Clip fraction (% d'actions clippées par PPO)

### 2. Formats de Sauvegarde

#### A. JSON (metrics.json)
Structure complète et lisible :
```json
{
  "episodes": [
    {
      "episode": 0,
      "reward": -21.0,
      "length": 1234,
      "global_step": 1234,
      "timestamp": "2025-10-21T15:46:22.123456"
    },
    ...
  ],
  "updates": [
    {
      "episode": 10,
      "total_loss": 0.123,
      "policy_loss": 0.045,
      "value_loss": 0.067,
      "entropy": 1.234,
      "clip_fraction": 0.12,
      "timestamp": "2025-10-21T15:46:23.123456"
    },
    ...
  ]
}
```

#### B. TensorBoard
Visualisation interactive en temps réel :
- Graphiques scalaires interactifs
- Comparaison multi-run
- Filtrage et lissage
- Export des données

#### C. Configuration (config.json)
Sauvegarde tous les hyperparamètres :
```json
{
  "base_model": "ppo_pong_20251021_110351/best_model.pth",
  "n_episodes": 500,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "clip_epsilon": 0.2,
  "gae_lambda": 0.95,
  "batch_size": 256,
  "epochs": 4,
  "timestamp": "20251021_154622",
  "device": "cpu"
}
```

## 📈 Utilisation du Système

### 1. Fine-tuning avec Tracking

```bash
# Lancement basique
python finetune_pong.py --episodes 500

# Avec paramètres personnalisés
python finetune_pong.py \
    --episodes 500 \
    --lr 1e-4 \
    --gamma 0.99 \
    --clip 0.2 \
    --plot

# Depuis un modèle spécifique
python finetune_pong.py \
    --model ppo_pong_20251021_110351/best_model.pth \
    --episodes 500
```

**Sorties automatiques:**
- Dossier `ppo_pong_finetuned_YYYYMMDD_HHMMSS/`
- Fichiers JSON avec toutes les métriques
- Logs TensorBoard en temps réel
- Checkpoints réguliers

### 2. Visualisation TensorBoard

```bash
# Lancer TensorBoard
tensorboard --logdir=ppo_pong_finetuned_20251021_154622/tensorboard

# Puis ouvrir dans le navigateur
# http://localhost:6006
```

**Métriques disponibles:**

| Catégorie | Métrique | Description |
|-----------|----------|-------------|
| Episode | Reward | Récompense par épisode |
| Episode | Length | Longueur en frames |
| Episode | Global_Step | Step cumulatif |
| Loss | Total | Somme des losses |
| Loss | Policy | Loss de la politique |
| Loss | Value | Loss du critique |
| Loss | Entropy | Entropie (exploration) |
| PPO | Clip_Fraction | % actions clippées |

### 3. Analyse Comparative

```bash
# Comparer tous les modèles
python analyze_performance.py --plot --report

# Comparer des modèles spécifiques
python analyze_performance.py \
    --models ppo_pong_A ppo_pong_B ppo_pong_C \
    --plot \
    --report \
    --output comparison.png
```

**Sorties:**
1. **Tableau comparatif** dans le terminal
2. **Graphiques de comparaison** (4 plots):
   - Récompenses lissées au fil du temps
   - Longueurs d'épisodes
   - Performance finale (bar chart)
   - Distribution des récompenses (boxplot)
3. **Rapport détaillé** en texte (`performance_report.txt`)

### 4. Accès Programmatique

```python
import json

# Charger les métriques
with open('ppo_pong_finetuned_XXXX/metrics.json', 'r') as f:
    metrics = json.load(f)

# Accéder aux épisodes
episodes = metrics['episodes']
rewards = [ep['reward'] for ep in episodes]

# Statistiques
avg_reward = np.mean(rewards[-100:])  # Moyenne des 100 derniers
print(f"Performance finale: {avg_reward:.2f}")

# Accéder aux updates
updates = metrics['updates']
policy_losses = [u['policy_loss'] for u in updates]
```

## 📊 Interprétation des Métriques

### Récompense (Reward)
- **Plage:** -21 à +21 (Pong)
- **Positif:** L'agent gagne plus de points
- **Négatif:** L'adversaire gagne plus de points
- **Objectif:** Maximiser la récompense moyenne

### Longueur d'Épisode
- **Plage:** Variable (500-3000+ frames typique)
- **Court:** Parties rapides (bonne/mauvaise perf)
- **Long:** Échanges équilibrés
- **Tendance:** Devrait se stabiliser

### Policy Loss
- **Rôle:** Mesure l'amélioration de la politique
- **Tendance:** Devrait décroître puis se stabiliser
- **Alerte:** Si augmente fortement → divergence

### Value Loss
- **Rôle:** Précision de l'estimation de valeur
- **Tendance:** Devrait décroître
- **Alerte:** Si reste élevé → critique mal calibré

### Entropie
- **Rôle:** Mesure d'exploration
- **Élevé:** Exploration (bon au début)
- **Faible:** Exploitation (normal en fin)
- **Alerte:** Si tombe trop vite → sous-exploration

### Clip Fraction
- **Rôle:** % d'actions clippées par PPO
- **Plage:** 0.0 à 1.0
- **Optimal:** 0.1 - 0.3
- **Trop haut (>0.5):** Updates trop agressifs
- **Trop bas (<0.05):** Updates trop conservateurs

## 🎯 Scénarios d'Analyse

### Scénario 1: Fine-tuning Réussi

**Signes:**
- ✅ Récompense moyenne augmente
- ✅ Variance se réduit progressivement
- ✅ Losses décroissent puis se stabilisent
- ✅ Clip fraction ~ 0.1-0.3
- ✅ Entropie décroît lentement

**Actions:**
- Continuer l'entraînement
- Sauvegarder le modèle actuel

### Scénario 2: Stagnation

**Signes:**
- ⚠️ Récompense plateau depuis 100+ épisodes
- ⚠️ Losses stables mais perf ne s'améliore pas
- ⚠️ Clip fraction très faible (<0.05)

**Actions:**
- Augmenter le learning rate
- Réduire clip_epsilon
- Augmenter l'entropie bonus (c2)

### Scénario 3: Instabilité

**Signes:**
- ❌ Récompense oscille fortement
- ❌ Losses augmentent
- ❌ Clip fraction très élevée (>0.5)

**Actions:**
- Réduire le learning rate
- Augmenter clip_epsilon
- Réduire la batch size
- Repartir d'un checkpoint antérieur

### Scénario 4: Overfitting

**Signes:**
- ⚠️ Losses continuent de baisser
- ⚠️ Performance test se dégrade
- ⚠️ Entropie proche de 0

**Actions:**
- Arrêter l'entraînement
- Utiliser un checkpoint antérieur
- Augmenter c2 (entropy bonus)

## 📁 Structure des Fichiers

```
ppo_pong_finetuned_20251021_154622/
├── metrics.json              # Métriques complètes
├── config.json               # Configuration d'entraînement
├── best_model.pth           # Meilleur modèle (avg 100 derniers)
├── checkpoint_ep50.pth      # Checkpoint épisode 50
├── checkpoint_ep100.pth     # Checkpoint épisode 100
├── ...
├── finetuning_results.png   # Graphiques (si --plot)
└── tensorboard/             # Logs TensorBoard
    ├── events.out.tfevents.xxx
    └── ...
```

## 🔧 Configuration Avancée

### Modifier la fréquence de sauvegarde

Dans `finetune_pong.py`:
```python
save_every=50    # Sauvegarder tous les 50 épisodes
log_every=10     # Logger tous les 10 épisodes
```

### Ajouter des métriques personnalisées

```python
# Dans PerformanceTracker.log_episode()
def log_episode(self, episode, reward, length, global_step, custom_metric=None):
    metric = {
        'episode': episode,
        'reward': reward,
        'length': length,
        'global_step': global_step,
        'custom': custom_metric,  # Nouvelle métrique
        'timestamp': datetime.now().isoformat()
    }
    self.episode_metrics.append(metric)
    
    if self.writer:
        self.writer.add_scalar('Custom/Metric', custom_metric, episode)
```

### Exporter vers d'autres formats

```python
import pandas as pd

# Charger les métriques
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

# Convertir en DataFrame
df_episodes = pd.DataFrame(metrics['episodes'])
df_updates = pd.DataFrame(metrics['updates'])

# Exporter en CSV
df_episodes.to_csv('episodes.csv', index=False)
df_updates.to_csv('updates.csv', index=False)

# Exporter en Excel
with pd.ExcelWriter('metrics.xlsx') as writer:
    df_episodes.to_excel(writer, sheet_name='Episodes', index=False)
    df_updates.to_excel(writer, sheet_name='Updates', index=False)
```

## 📊 Workflow Recommandé

### 1. Lancement
```bash
python finetune_pong.py --episodes 500 --lr 1e-4
```

### 2. Monitoring en temps réel
```bash
# Terminal 2
tensorboard --logdir=ppo_pong_finetuned_*/tensorboard
```

### 3. Analyse périodique
```bash
# Toutes les 100 épisodes, vérifier:
python analyze_performance.py --plot
```

### 4. Ajustements
- Si stagnation → modifier hyperparamètres
- Si bon progrès → continuer
- Si divergence → rollback checkpoint

### 5. Validation finale
```bash
# Tester le meilleur modèle
python play_pong.py --games 20

# Analyse comparative
python analyze_performance.py --plot --report
```

## 🎓 Conseils d'Optimisation

### Pour accélérer l'entraînement
- Augmenter batch_size (256 → 512)
- Réduire update_every (2048 → 1024)
- Utiliser GPU si disponible

### Pour améliorer la stabilité
- Réduire learning_rate (1e-4 → 5e-5)
- Augmenter clip_epsilon (0.2 → 0.3)
- Augmenter epochs (4 → 6)

### Pour favoriser l'exploration
- Augmenter c2 entropy bonus (0.01 → 0.02)
- Démarrer avec learning_rate plus élevé

## 🔍 Debugging

### Metrics non sauvegardés
```python
# Vérifier que tracker.save_metrics() est appelé
tracker.save_metrics()

# Forcer la sauvegarde périodique
if (episode + 1) % 50 == 0:
    tracker.save_metrics()
```

### TensorBoard ne montre rien
```bash
# Vérifier le chemin
ls ppo_pong_finetuned_*/tensorboard/

# Forcer le flush
if self.writer:
    self.writer.flush()
```

### Métriques incohérentes
```python
# Ajouter des assertions
assert -21 <= reward <= 21, f"Reward hors limites: {reward}"
assert length > 0, f"Longueur invalide: {length}"
```

## 📚 Ressources

- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **PyTorch Docs**: https://pytorch.org/docs/

## ✅ Checklist

Avant chaque session d'entraînement :
- [ ] Vérifier que le bon modèle de base est chargé
- [ ] Confirmer les hyperparamètres
- [ ] Vérifier l'espace disque disponible
- [ ] Lancer TensorBoard pour monitoring
- [ ] Définir un critère d'arrêt clair

Après chaque session :
- [ ] Sauvegarder les métriques (`tracker.save_metrics()`)
- [ ] Générer les graphiques de comparaison
- [ ] Documenter les observations
- [ ] Backup des meilleurs modèles
- [ ] Commit sur Git avec description

---

**Dernière mise à jour:** 21 octobre 2025
