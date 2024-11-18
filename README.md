# Sub-Catan AECEnv
Sub-Catan is a simplified implementation of the board game Catan, designed as a multi-agent reinforcement learning environment
compatible with the PettingZoo library. It makes it possible to train models compatible with the gym API in this turn-based multi-
agent setting, with room to expand on the gamerules and actions!

!!! The environment is located at branch 4-Agents-interface-torch, main branch is only a playable version !!!

## Table of Contents
- Features
- Requirements
- Installation
- Project Structure
- Usage
- Environment Details
- Examples
- License
- Contact Information

## Features
- Simplified Catan game environment for reinforcement learning.
- Compatible with the PettingZoo library for multi-agent environments.
- Supports both settlement phase and normal phase gameplay.
- Customizable rendering modes: 'human', 'rgb_array', or None.
- Easy integration with popular RL algorithms in PyTorch.
- Logging and debugging support with TensorBoard.

## Requirements
Python 3.11.9 (at least 3.9 or higher, this is not tested)

numpy==2.1.3

gymnasium==1.0.0

pettingzoo==1.24.3

torch==2.5.1

tensorboard==2.18.0

pygame==2.6.1

imageio==2.36.0


linux distro is highly recommended, a WSL installation of Ubuntu 22.04 with a miniconda virtual environment 
was used for development.

## Installation

1.Clone the Repository:

    git clone https://github.com/yourusername/sub-catan.git
    cd sub-catan

2.Create a virtual environment (optional but recommended):

    conda create --name <venv_name> python=3.11.9
    conda activate <venv_name>

3.Install required packages:

    pip install --upgrade pip
    pip install -r requirements.txt

or install packages individually:

    pip install numpy==2.1.3 gymnasium==1.0.0 pettingzoo==1.24.3 torch==2.5.1 tensorboard==2.18.0 pygame==2.6.1 imageio==2.36.0

## Project Structure

- main.py: The main entry point of the project. Contains functions to run all runnable modes of the project, game loop
train models, pretrain settlement phase, and evaluating trained agents
- environment/: Contains the custom environment implementation for Sub-Catan
-    - CatanEnv.py: Defines the CatanEnv class, the PettingZoo AECEnv main environment
-    - CustomAgentSelector.py: Custom agent selector for managing agent turn order in the settle phase.
- game/: Contains game logic and components:
-    - GameBoard.py: Defines the game board structure.
-    - GameManager_env.py: Manages the game state updates and agent actions.
-    - GameRules.py: Contains the game rules for validation of player moves.
-    - Player.py: Represents player states
-    - House.py / City.py /Road.py: represent game pieces.
- models/: Contains models and training scripts
-    - Trainer.py: Conatins training routines for agents.
-    - Evaluator.py: Functions to evaluate trained agents.
- assets/: Contains utility classes.
-    - Console.py / PrintConsole.py: Utilities for logging and printing game events.
- gameloop/: Contains the game loop for playtesting the game yourself.
-    - Gameloop.py
- requirements.txt: List of all Python dependencies

## Usage

There is currently 4 configurations in the main.py to entry different scripts of the project:

### Running configurations
1. Playtest the game yourself: Uncomment GameLoop().main()
2. Pretrain agents in settlephase only: Uncomment pretrain_settlement_phase()
3. Train agents in either normal phase or whole game (look in main.py documentation for instructions)
4. Evaluate trained agents in either normal phase or whole game.

### Monitor training
Training logs and metrics are saved using TensorBoard. 

tensorboard --logdir run_logs/

## Environment Details
### Action space
The action space is a Discrete space of size determined by the number of possible actions:

- Pass: End the current player's turn
- Roll Dice: Roll the dice to produce resources
- Build Settlement: Place a settlement on an unoccupied vertex.
- Build City: Upgrade from a settlement to a city
- Build Roads: Place a road on an unoccupied edge.

The total action space size is calculated using the number of vertices and edges:

action_space_size = 2 + 2 * num_vertecies + num_edges

- 2 for pass and roll dice actions
- num_vertecies for settlement actions
- num_vertecies for city upgrades
- num_edges for road placement

### Observation Space
The observation space is a Dict containing:
- action_mask: A binary vector indication valid actions at the current state
- observation: A flattened vector representing:
-    - Board State: Includes information about vertices, edges, and tiles. 
-    -    - Vertex states (houses, cities)
-    -    - Edge states (roads)
-    -    - Tile states (resources, numbers)
-    - Player State: Players resources, remaining pieces and victory points
-    - Enemy State: Aggregated resources and victory points of other players.

### Reward function
Rewards is given based on the player's actions and game progress:

- Positive for:
-    - Winning the game, or placement (first to last)
-    - Building settlements and cities (actions leadning to victory points)
-    - Rolling dice (to encurage partaking rather than just passing turns)
- Negative for:
-    - Invalid actions
-    - Passing turns (encurage progress)

In the current iteration, the point is to encurage the agents to gain victory points as fast as possible.

## Contact Information
Feel free to reach out if anything:

- Email: henritf@ntnu.no
- GitHub Issues: [Issues](https://github.com/HenrikFredriksen/sub-catan/issues)





