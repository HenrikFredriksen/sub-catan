
class GameRules:
    def __init__(self, game_board):
        self.game_board = game_board
        
    def is_valid_house_placement(self, vertex):
        if vertex.house is not None:
            return False
        return True