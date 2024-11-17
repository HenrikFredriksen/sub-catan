import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from game.GameBoard import GameBoard
from game.GameRules import GameRules
from game.GameManager_env import GameManager
from game.Player import Player
from assets.PrintConsole import PrintConsole

import os
import pickle

def generate_valid_board_with_settlements(players, max_attempts=100):
    
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}")
        
        game_board = GameBoard()
        game_board.generate_board(board_radius=2)
        game_rules = GameRules(game_board)
        
        console = PrintConsole()
        game_manager = GameManager(game_board, game_rules, players, console)
        
        success = game_manager.simulate_settle_phase()
        
        if success:
            print("Valid board found!")
            return game_board
        else:
            print("Invalid board")
            
    raise Exception("Could not generate a valid board after {max_attempts} attempts")

def generate_and_save_boards(num_boards, players, output_dir, max_attempts=100):
    os.makedirs(output_dir, exist_ok=True)
    generated_boards = 0
    while generated_boards < num_boards:
        print(f"Generated {generated_boards + 1}/{num_boards} boards")
        
        game_board = GameBoard()
        game_board.generate_board(board_radius=2)
        game_rules = GameRules(game_board)
        console = PrintConsole()
        game_manager = GameManager(game_board, game_rules, players, console)
        
        success = game_manager.simulate_settle_phase()
        
        if success:
            print("Valid board found!")
            game_board.clean_for_serialization()
            
            board_filename = f"game_board_{generated_boards}.pkl"
            board_filepath = os.path.join(output_dir, board_filename)
            with open(board_filepath, 'wb') as f:
                pickle.dump(game_board, f)
            print(f"Board saved to {board_filepath}")
            generated_boards += 1
        else:
            print("Invalid board configuration, retrying ..")
            
def main():
    players = [
        Player(player_id=0, color=(255, 0, 0)),
        Player(player_id=1, color=(0, 0, 255)),
        Player(player_id=2, color=(20, 220, 20)),
        Player(player_id=3, color=(255, 165, 0))
    ]
    
    generate_and_save_boards(num_boards=1000, players=players, output_dir='normal_phase_boards')
    
if __name__ == '__main__':
    main()