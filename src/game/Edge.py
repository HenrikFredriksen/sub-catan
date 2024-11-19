'''
This class represents an edge in the graph. It has two vertices and a road.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Edge:
    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.road = None
        