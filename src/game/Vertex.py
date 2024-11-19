'''
The Vertex class represents a vertex on the game board.
The vertex has a position, and a list of neighbors.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Vertex:
    def __init__(self, position):
        self.position = position
        self.house = None
        self.city = None
        self.neighbors = []
        
    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)