import HexCoordinate as pos
import numpy as np

class House:
    def __init__(self, vertex, player):
        self.vertex = vertex
        self.player = player
        
    
    def get_player(self):
        return self.player
    