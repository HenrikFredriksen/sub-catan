
class GameRules:
    def __init__(self, game_board):
        self.game_board = game_board
        
    def is_valid_house_placement(self, vertex, player, phase):
        if vertex.house is not None:
            return False
        
        # check if there is a house on any of the neighbors
        for neighbor in vertex.neighbors:
            if neighbor.house is not None or neighbor.city is not None:
                return False
    
        if phase == 'normal_phase':
            for neighbor in vertex.neighbors:
                edge_key = tuple(sorted([vertex.position, neighbor.position]))
                edge = self.game_board.edges.get(edge_key)
                if edge and edge.road and edge.road.player == player:
                    return True
            return False
        
        return True
    
    def is_valid_road_placement(self, edge, player, phase, last_placed_house_vertex=None):
        if edge.road is not None:
            return False
        
        if phase == 'settle_phase':
            if (last_placed_house_vertex and 
                (edge.vertex1 == last_placed_house_vertex or 
                 edge.vertex2 == last_placed_house_vertex)):
                return True
            return False
        
        elif phase == 'normal_phase':
            if edge.vertex1.house and edge.vertex1.house.player == player:
                return True
            if edge.vertex2.house and edge.vertex2.house.player == player:
                return True

            for neighbor in edge.vertex1.neighbors:
                if (neighbor != edge.vertex2 and 
                    self.game_board.edges.get(tuple(sorted([edge.vertex1.position, 
                                                            neighbor.position])))):
                    if (self.game_board.edges[tuple(sorted([edge.vertex1.position, 
                                                            neighbor.position]))].road and 
                        self.game_board.edges[tuple(sorted([edge.vertex1.position, 
                                                            neighbor.position]))].road.player == player):
                        return True
            for neighbor in edge.vertex2.neighbors:
                if (neighbor != edge.vertex1 and 
                    self.game_board.edges.get(tuple(sorted([edge.vertex2.position, 
                                                            neighbor.position])))):
                    if (self.game_board.edges[tuple(sorted([edge.vertex2.position, 
                                                            neighbor.position]))].road and 
                        self.game_board.edges[tuple(sorted([edge.vertex2.position, 
                                                            neighbor.position]))].road.player == player):
                        return True
            return False
    
    def starting_settlement_bonus(self, vertex):
        if vertex.house is None:
            return False
        
    def is_valid_city_placement(self, vertex, player):
        return vertex.house is not None and vertex.house.player == player