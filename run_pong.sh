#!/bin/bash

# Script de démarrage rapide pour le projet PPO Pong

echo "🏓 Projet PPO Pong - Menu Principal"
echo "======================================"
echo ""
echo "Que voulez-vous faire ?"
echo ""
echo "1. Entraîner un nouveau modèle"
echo "2. Fine-tuner un modèle existant"
echo "3. Jouer contre l'agent (mode auto)"
echo "4. Jouer contre l'agent (mode interactif)"
echo "5. Analyser les performances"
echo "6. Lancer TensorBoard"
echo "7. Quitter"
echo ""
read -p "Votre choix (1-7): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Lancement de l'entraînement..."
        python pong.py
        ;;
    2)
        echo ""
        read -p "Nombre d'épisodes (défaut: 500): " episodes
        episodes=${episodes:-500}
        read -p "Learning rate (défaut: 1e-4): " lr
        lr=${lr:-1e-4}
        echo ""
        echo "🔄 Lancement du fine-tuning..."
        python finetune_pong.py --episodes $episodes --lr $lr --plot
        ;;
    3)
        echo ""
        read -p "Nombre de parties (défaut: 3): " games
        games=${games:-3}
        echo ""
        echo "🎮 Lancement du mode spectateur..."
        python play_pong.py --games $games
        ;;
    4)
        echo ""
        read -p "Nombre de parties (défaut: 3): " games
        games=${games:-3}
        echo ""
        echo "🎮 Lancement du mode interactif..."
        python play_interactive_pong.py --games $games
        ;;
    5)
        echo ""
        echo "📊 Analyse des performances..."
        python analyze_performance.py --plot --report
        ;;
    6)
        echo ""
        echo "Dossiers disponibles:"
        ls -d ppo_pong_* 2>/dev/null || echo "  Aucun dossier trouvé"
        echo ""
        read -p "Chemin vers le dossier (ou entrée pour le plus récent): " logdir
        
        if [ -z "$logdir" ]; then
            logdir=$(ls -td ppo_pong_*/ 2>/dev/null | head -1)
            if [ -z "$logdir" ]; then
                echo "❌ Aucun dossier de logs trouvé!"
                exit 1
            fi
        fi
        
        tensorboard_dir="${logdir}tensorboard"
        if [ ! -d "$tensorboard_dir" ]; then
            echo "❌ Dossier TensorBoard non trouvé: $tensorboard_dir"
            exit 1
        fi
        
        echo ""
        echo "📈 Lancement de TensorBoard..."
        echo "Ouvrez votre navigateur sur: http://localhost:6006"
        echo "Appuyez sur Ctrl+C pour arrêter"
        echo ""
        tensorboard --logdir="$tensorboard_dir"
        ;;
    7)
        echo ""
        echo "👋 Au revoir!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Choix invalide!"
        exit 1
        ;;
esac
