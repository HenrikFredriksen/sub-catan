class Vertex:
    def __init__(self, position):
        self.position = position
        self.house = None
        self.city = None
        self.neighbors = []
        
    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)