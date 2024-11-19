'''
The HexCoordinate class is a class that represents a hexagonal coordinate in a hex grid.
It has a column (q) and a row (r) attribute, and methods for getting neighbors and calculating distance.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class HexCoordinate:
    def __init__(self, q, r):
        self.q = q # column
        self.r = r # row
        
    def __eq__(self, other):
        return self.q == other.q and self.r == other.r
    
    def __hash__(self):
        return hash((self.q, self.r))
    
    def get_neighbors(self):
        # List of acial directions in a hex grid
        directions = [
            HexCoordinate(1, 0), HexCoordinate(1, -1), HexCoordinate(0, -1),
            HexCoordinate(-1, 0), HexCoordinate(-1, 1), HexCoordinate(0, 1)
        ]
        neighbors = []
        for direction in directions:
            neighbor = HexCoordinate(self.q + direction.q, self.r + direction.r)
            neighbors.append(neighbor)
        return neighbors
    
    def distance(self, other):
        return (abs(self.q - other.q) 
                + abs(self.q + self.r - other.q - other.r) 
                + abs(self.r - other.r)) // 2