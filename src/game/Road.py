'''
The Road class represents a road on the game board.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Road:
    def __init__(self, vertex1, vertex2, player):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.player = player
        
    def get_player(self):
        return self.player