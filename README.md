# AI-Powered-Distributed-Snake-Game-Simulation
An advanced AI-driven Snake Game Simulation built with PyTorch Distributed Data Parallel (DDP) and Deep Q-Learning (DQN), designed to leverage the power of distributed systems and reinforcement learning. This project brings the classic Snake game into the realm of modern AI, allowing the snake to learn optimal movements in a distributed, multi-device environment.

Distributed Training: The simulation runs across multiple devices in a single network, with each device contributing to the learning process.
Deep Q-Learning (DQN): The snake learns to navigate and grow through reinforcement learning, optimizing its moves to avoid collisions and reach food efficiently.
ROCm & PyTorch DDP Support: Fully compatible with AMD GPUs via ROCm, taking advantage of GPU-accelerated training.
Arcade-Based Visualization: The simulation includes a visual interface powered by the Arcade library, allowing real-time observation of the AI's learning progress.

PyTorch: For building and training the deep learning model.
Distributed Data Parallel (DDP): To handle distributed training across multiple devices.
ROCm: Optimized for AMD GPUs for faster processing.
Arcade: For creating an interactive, visually appealing game environment.

    pip install -r requirements.txt

    Set Environment Variables (on each device):


  export MASTER_ADDR="192.168.1.1"  # Set the IP of your master node
  export MASTER_PORT="29500"         # Port for distributed communication
  export WORLD_SIZE=4                # Number of devices in your network
  export RANK=<rank>                 # Unique rank for each device
