import numpy as np


'''
The House class represents a house/settlement on the game board.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class House:
    def __init__(self, vertex, player):
        self.vertex = vertex
        self.player = player
        
    
    def get_player(self):
        return self.player
    