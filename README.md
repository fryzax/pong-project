PPO Pong Reinforcement Learning Project

This project implements and fine-tunes a Proximal Policy Optimization (PPO) agent to learn and play the Atari Pong game using Gymnasium and PyTorch.

1. Project Overview

The goal of this project is to train an intelligent agent capable of mastering Pong through reinforcement learning.
The agent learns directly from visual inputs (game frames) and improves its strategy through continuous interaction with the environment.

2. Repository Structure 
PONG-PROJECT/
│
├── ppo_pong_*/                      # Training and fine-tuning session folders (saved models, logs, plots)
├── pong.py                          # Main training script from scratch
├── finetune_pong.py                 # Fine-tuning script using the pretrained PPO model
├── evaluate_pong.py                 # Evaluation of trained models (without training)
├── play_pong.py                     # Script to visualize the trained agent playing Pong
├── play_interactive_pong.py         # Interactive mode to play against the trained agent
├── analyze_performance.py           # Analyze and plot performance metrics
├── compare.py                       # Compare results between different training runs
├── python_plot.py                   # Utility script for visualizations
├── best_try.py                      # Script for testing the best model configuration
├── ppo_pong_final_best.pth          # Final saved PPO model
├── run_pong.sh                      # Shell script for automated execution
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

3. Installation 
git clone <repo_link>
cd PONG-PROJECT
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

4. Usage
Train PPO from scratch
python pong.py

Fine-tune a pretrained model
python finetune_pong.py --model ppo_pong_final_best.pth --episodes 500

Evaluate a trained agent
python evaluate_pong.py

Watch the agent play Pong
python play_pong.py

Compare or analyze results
python analyze_performance.py

5. Results

PPO successfully learns to play Pong from raw pixels.

Fine-tuning improves stability and average rewards.

Training logs, curves, and model checkpoints are automatically saved in timestamped folders (e.g., ppo_pong_finetuned_20251027_151334).

6. Requirements

Python 3.10+

PyTorch

Gymnasium with ALE (Atari Learning Environment)

NumPy, Matplotlib, SciPy, TensorBoard

Install all dependencies with:

pip install -r requirements.txt


7. Authors

Mathieu Souesme
Antonin Arroyo

Master 2 Data for Business, Albert School
Project: Reinforcement Learning — PPO Agent for Pong