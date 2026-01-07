MiniGrid Curriculum Learning Project

This project trains an AI agent to solve complex MiniGrid environments using a Phased Forward Curriculum Learning strategy. The agent learns to navigate large mazes (up to 12 rooms), handle colored keys and locks, and avoid lava hazards.

📦 Installation

This project requires Python 3.8+ and the following libraries:

pip install gymnasium minigrid stable-baselines3 torch tensorboard tqdm


🧠 Training Phases

The training process is divided into four distinct phases to incrementally build the agent's skills.

1. Phase 1: Basic Navigation

Trains the agent to find the goal in empty rooms (up to 8 rooms) to solve the sparse reward problem.

python train.py


2. Phase 2: Key-Door Skills

Fine-tunes the agent in MiniGrid-DoorKey-8x8-v0 to learn specific "Pickup" and "Toggle" actions required to interact with objects.

python train_key.py


3. Phase 3: Integration (Multicolor)

Combines navigation and key usage in a custom CorredorMulticolor environment (up to 12 rooms).

python train_integration.py


4. Phase 4: Final Complexity (Smart Lava)

Adds lethal hazards using a BFS-based "Smart Lava" algorithm that protects critical paths, forcing the agent to be precise.

python train_lava.py


🎮 Visualization

To watch the trained agent perform in the final environment:

python visualize.py


This script renders 3 episodes of the LavaCorridorFinal environment.

📂 File Description

Filename

Description

train.py

Phase 1. Curriculum for basic navigation (2, 6, 8 rooms).

train_key.py

Phase 2. Specialized training for DoorKey tasks.

train_integration.py

Phase 3. Custom environment CorredorMulticolor implementation.

train_lava.py

Phase 4. Final training with CorredorLavaSmart.

visualize.py

Inference script for human evaluation and rendering.