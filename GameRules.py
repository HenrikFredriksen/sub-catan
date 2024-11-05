
class GameRules:
    def __init__(self, game_board):
        self.game_board = game_board
        
    def is_valid_house_placement(self, vertex):
        if vertex.house is not None:
            return False
        
        # check if there is a house on any of the neighbors
        for neighbor in vertex.neighbors:
            if neighbor.house is not None:
                return False
        return True
    
    def is_valid_road_placement(self, edge, player):
        if edge.road is not None:
            return False
        
        if edge.vertex1.house and edge.vertex1.house.player == player:
            return True
        if edge.vertex2.house and edge.vertex2.house.player == player:
            return True
        
        for neighbor in edge.vertex1.neighbors:
            if neighbor != edge.vertex2 and self.game_board.edges.get(tuple(sorted([edge.vertex1.position, neighbor.position]))):
                if self.game_board.edges[tuple(sorted([edge.vertex1.position, neighbor.position]))].road and self.game_board.edges[tuple(sorted([edge.vertex1.position, neighbor.position]))].road.player == player:
                    return True
        for neighbor in edge.vertex2.neighbors:
            if neighbor != edge.vertex1 and self.game_board.edges.get(tuple(sorted([edge.vertex2.position, neighbor.position]))):
                if self.game_board.edges[tuple(sorted([edge.vertex2.position, neighbor.position]))].road and self.game_board.edges[tuple(sorted([edge.vertex2.position, neighbor.position]))].road.player == player:
                    return True
        return False