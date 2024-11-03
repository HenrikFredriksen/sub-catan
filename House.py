import HexCoordinate as pos
import numpy as np

class House:
    def __init__(self, vertex, player):
        self.player = player
        self.vertex = vertex
        
    
    def get_player(self):
        return self.player
    