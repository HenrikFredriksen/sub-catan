'''
City class is used to represent a city in the game. It has a vertex and a player.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class City:
    def __init__(self, vertex, player):
        self.vertex = vertex
        self.player = player
        
    def get_player(self):
        return self.player