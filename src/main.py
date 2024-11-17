import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Evaluator import eval_trained_agents
from models.Trainer import train_normal_phase_loaded_board, pretrain_settlement_phase
from gameloop.GameLoop import main

if __name__ == "__main__":
    '''
    this is the main entry point for the project.
    uncomment the function you want to run.
    '''
    
    #main() # this is the main game loop, not used for model training, just for playing the game.
    
    
    #pretrain_settlement_phase() # Train models only in settle phase of the game
    
    
    #train_normal_phase_loaded_board() # Train models in normal phase of the game with a loaded board state
    
    # Evaluate the trained agents, change render_mode in Evaluator.py
    # render_mode = 'rgb_array' or 'human'
    num_evals = 10
    for _ in range(num_evals):
        eval_trained_agents()