#Python script to compare performance metrics of multiple models we trained

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


def load_metrics(model_dir):
    """Load metrics from a model directory"""
    metrics_file = os.path.join(model_dir, "metrics.json")
    config_file = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    config = None
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return {
        'metrics': metrics,
        'config': config,
        'model_dir': model_dir
    }


def compute_statistics(episodes, window=100):
    """Compute statistics from episode data"""
    rewards = [e['reward'] for e in episodes]
    lengths = [e['length'] for e in episodes]
    
    stats = {
        'total_episodes': len(episodes),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'avg_length': np.mean(lengths),
        'final_avg_reward': np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards),
        'final_std_reward': np.std(rewards[-window:]) if len(rewards) >= window else np.std(rewards),
    }
    
    return stats


def compare_models(model_dirs):
    """Compare multiple models and generate report"""
    print("\n" + "="*80)
    print("ANALYSE COMPARATIVE DES MODÈLES PPO PONG")
    print("="*80 + "\n")
    
    models_data = []
    
    for model_dir in model_dirs:
        data = load_metrics(model_dir)
        if data is None:
            print(f"⚠️  Pas de métriques trouvées pour: {model_dir}")
            continue
        
        episodes = data['metrics']['episodes']
        stats = compute_statistics(episodes)
        
        models_data.append({
            'name': os.path.basename(model_dir),
            'dir': model_dir,
            'stats': stats,
            'config': data['config'],
            'episodes': episodes
        })
    
    if not models_data:
        print("❌ Aucun modèle valide trouvé!")
        return
    
    # Sort by final average reward
    models_data.sort(key=lambda x: x['stats']['final_avg_reward'], reverse=True)
    
    # Print comparison table
    print(f"{'Modèle':<40} {'Épisodes':>10} {'Récompense Finale':>20} {'Max':>10}")
    print("-" * 80)
    
    for model in models_data:
        name = model['name'][:38]
        stats = model['stats']
        print(f"{name:<40} {stats['total_episodes']:>10} "
              f"{stats['final_avg_reward']:>10.2f} ± {stats['final_std_reward']:>5.2f} "
              f"{stats['max_reward']:>10.2f}")
    
    print("\n" + "="*80)
    print("DÉTAILS DES PERFORMANCES")
    print("="*80 + "\n")
    
    for i, model in enumerate(models_data, 1):
        print(f"\n{i}. {model['name']}")
        print("-" * 80)
        stats = model['stats']
        print(f"  📊 Statistiques globales:")
        print(f"     • Total d'épisodes: {stats['total_episodes']}")
        print(f"     • Récompense moyenne: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"     • Récompense min/max: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"     • Longueur moyenne: {stats['avg_length']:.2f}")
        print(f"  🎯 Performance finale (100 derniers épisodes):")
        print(f"     • Récompense: {stats['final_avg_reward']:.2f} ± {stats['final_std_reward']:.2f}")
        
        if model['config']:
            print(f"  ⚙️  Configuration:")
            config = model['config']
            if 'learning_rate' in config:
                print(f"     • Learning rate: {config['learning_rate']}")
            if 'gamma' in config:
                print(f"     • Gamma: {config['gamma']}")
            if 'clip_epsilon' in config:
                print(f"     • Clip epsilon: {config['clip_epsilon']}")
            if 'base_model' in config:
                print(f"     • Modèle de base: {config['base_model']}")
    
    print("\n" + "="*80 + "\n")
    
    return models_data


def plot_comparison(models_data, save_path=None):
    """Plot comparison of multiple models"""
    if not models_data:
        return
    
    n_models = len(models_data)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Smooth function
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Rewards over time
    ax = axes[0, 0]
    for i, model in enumerate(models_data):
        episodes = model['episodes']
        rewards = [e['reward'] for e in episodes]
        episode_nums = [e['episode'] for e in episodes]
        
        smoothed = smooth(rewards)
        ax.plot(episode_nums[:len(smoothed)], smoothed, 
                color=colors[i], label=model['name'][:30], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title('Comparaison des Récompenses')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    #Episode lengths
    ax = axes[0, 1]
    for i, model in enumerate(models_data):
        episodes = model['episodes']
        lengths = [e['length'] for e in episodes]
        episode_nums = [e['episode'] for e in episodes]
        
        smoothed = smooth(lengths)
        ax.plot(episode_nums[:len(smoothed)], smoothed,
                color=colors[i], label=model['name'][:30], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length (smoothed)')
    ax.set_title('Comparaison des Longueurs d\'Épisodes')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    #Final performance
    ax = axes[1, 0]
    names = [m['name'][:20] for m in models_data]
    final_rewards = [m['stats']['final_avg_reward'] for m in models_data]
    final_stds = [m['stats']['final_std_reward'] for m in models_data]
    
    bars = ax.bar(range(n_models), final_rewards, yerr=final_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Récompense Moyenne (100 derniers)')
    ax.set_title('Performance Finale')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val, std) in enumerate(zip(bars, final_rewards, final_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    #Distribution of rewards 
    ax = axes[1, 1]
    reward_data = []
    for model in models_data:
        episodes = model['episodes']
        rewards = [e['reward'] for e in episodes[-100:]]  # Last 100 episodes
        reward_data.append(rewards)
    
    bp = ax.boxplot(reward_data, labels=[m['name'][:20] for m in models_data],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels([m['name'][:20] for m in models_data], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Récompense')
    ax.set_title('Distribution des Récompenses (100 derniers épisodes)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Graphique de comparaison sauvegardé: {save_path}")
    
    plt.show()


def generate_report(model_dirs, output_file='performance_report.txt'):
    """Generate a detailed text report"""
    models_data = []
    
    for model_dir in model_dirs:
        data = load_metrics(model_dir)
        if data:
            episodes = data['metrics']['episodes']
            stats = compute_statistics(episodes)
            models_data.append({
                'name': os.path.basename(model_dir),
                'dir': model_dir,
                'stats': stats,
                'config': data['config'],
                'episodes': episodes
            })
    
    if not models_data:
        print("❌ Aucun modèle valide trouvé!")
        return
    
    models_data.sort(key=lambda x: x['stats']['final_avg_reward'], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ANALYSE DES PERFORMANCES - PPO PONG\n")
        f.write(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Nombre de modèles analysés: {len(models_data)}\n\n")
        
        f.write("CLASSEMENT PAR PERFORMANCE FINALE\n")
        f.write("-"*80 + "\n")
        for i, model in enumerate(models_data, 1):
            stats = model['stats']
            f.write(f"{i}. {model['name']}\n")
            f.write(f"   Récompense finale: {stats['final_avg_reward']:.2f} ± {stats['final_std_reward']:.2f}\n")
            f.write(f"   Récompense max: {stats['max_reward']:.2f}\n")
            f.write(f"   Épisodes: {stats['total_episodes']}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DÉTAILS COMPLETS\n")
        f.write("="*80 + "\n\n")
        
        for i, model in enumerate(models_data, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"MODÈLE {i}: {model['name']}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("STATISTIQUES GLOBALES\n")
            f.write("-"*80 + "\n")
            stats = model['stats']
            f.write(f"  Total d'épisodes: {stats['total_episodes']}\n")
            f.write(f"  Récompense moyenne: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}\n")
            f.write(f"  Récompense min: {stats['min_reward']:.2f}\n")
            f.write(f"  Récompense max: {stats['max_reward']:.2f}\n")
            f.write(f"  Longueur moyenne: {stats['avg_length']:.2f}\n\n")
            
            f.write("PERFORMANCE FINALE (100 derniers épisodes)\n")
            f.write("-"*80 + "\n")
            f.write(f"  Récompense: {stats['final_avg_reward']:.2f} ± {stats['final_std_reward']:.2f}\n\n")
            
            if model['config']:
                f.write("CONFIGURATION\n")
                f.write("-"*80 + "\n")
                for key, value in model['config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    print(f"\n📄 Rapport détaillé sauvegardé: {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyser les performances des modèles PPO Pong")
    parser.add_argument('--models', nargs='+', default=None,
                       help='Chemins vers les dossiers des modèles (par défaut: tous les ppo_pong_*)')
    parser.add_argument('--plot', action='store_true',
                       help='Générer des graphiques de comparaison')
    parser.add_argument('--report', action='store_true',
                       help='Générer un rapport texte détaillé')
    parser.add_argument('--output', type=str, default='performance_comparison.png',
                       help='Nom du fichier de sortie pour les graphiques')
    
    args = parser.parse_args()
    
    # Find model directories
    if args.models:
        model_dirs = args.models
    else:
        model_dirs = glob.glob('ppo_pong_*')
        if not model_dirs:
            print("❌ Aucun dossier de modèle trouvé!")
            print("Recherche de dossiers commençant par 'ppo_pong_'")
            return
    
    print(f"\n🔍 Analyse de {len(model_dirs)} modèle(s)...\n")
    
    # Compare models
    models_data = compare_models(model_dirs)
    
    if not models_data:
        return
    
    # Generate plots
    if args.plot:
        print("\n📊 Génération des graphiques...")
        plot_comparison(models_data, save_path=args.output)
    
    # Generate report
    if args.report:
        print("\n📄 Génération du rapport...")
        generate_report(model_dirs)
    
    print("\n✅ Analyse terminée!\n")


if __name__ == "__main__":
    main()
