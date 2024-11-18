import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Evaluator import eval_trained_agents
from models.Trainer import train, pretrain_settlement_phase
from gameloop.GameLoop import GameLoop

if __name__ == "__main__":
    '''
    this is the main entry point for the project.
    uncomment the function you want to run.
    
    main() - this is the main game loop, not used for model training, just for playing the game.
    
    train() - Train models, takes optional argument: 
        - gamestate='normal_phase' or 'settle_phase'
        
    pretrain_settlement_phase() - Train models only in settle phase of the game
    
    eval_trained_agents() - Evaluate the trained agents, takes optional arguments: 
        - render_mode='rgb_array' or 'human'
        - gamestate='normal_phase' or 'settle_phase'
    '''
    
    #GameLoop().main()
    
    #pretrain_settlement_phase()
    
    # Change gamestate 'settle_phase' to run whole game training
    # Change gamestate 'normal_phase' to run normal phase training with loaded board
    #train(n_episodes=1) # Train models in normal phase of the game with a loaded board state
    

    # render_mode = 'rgb_array' or 'human'
    # gamestate = 'normal_phase' or 'settle_phase'
    num_evals = 3
    for _ in range(num_evals):
        eval_trained_agents(render_mode='human', gamestate='normal_phase') # Evaluate the trained agents
