from game.House import House
from game.Road import Road
import numpy as np

class GameManager:
    def __init__(self, game_board, game_rules, players):
        self.game_board = game_board
        self.game_rules = game_rules
        self.players = players
        self.current_player_index = 0
        self.game_over = False
        
    @property
    def current_player(self):
        return self.players[self.current_player_index]
    
    def handle_click(self, mouse_pos):
        print(f"Mouse clicked at: {mouse_pos}")
        proximity_radius = self.game_board.hex_size / 4
        
        nearest_vertex = self.game_board.find_nearest_vertex(mouse_pos, proximity_radius)
        nearest_edge = self.game_board.find_nearest_edge(mouse_pos, proximity_radius)
        
        if nearest_vertex and nearest_edge:
            vertex_distance = np.hypot(mouse_pos[0] - nearest_vertex.position[0], mouse_pos[1] - nearest_vertex.position[1])
            edge_distance = np.hypot(mouse_pos[0] - (nearest_edge.vertex1.position[0] + nearest_edge.vertex2.position[0]) // 2, 
                                     mouse_pos[1] - (nearest_edge.vertex1.position[1] + nearest_edge.vertex2.position[1]) // 2)
            if vertex_distance < edge_distance:
                self.place_house(nearest_vertex)
            else:
                self.place_road(nearest_edge)
        elif nearest_vertex:
            self.place_house(nearest_vertex)
        elif nearest_edge:
            self.place_road(nearest_edge)
        else:
            print("No vertex or edge found")
            

    def place_house(self, vertex):
        print(f"Nearest vertex: {vertex.position}")
        if self.current_player.can_build_settlement() and self.game_rules.is_valid_house_placement(vertex):
            house = House(vertex=vertex, player=self.current_player)
            vertex.house = house
            self.current_player.settlements -= 1
            print(f"Placed house at {vertex.position}")
        else:
            print("Invalid House placement")
            
    def place_road(self, edge):
        print(f"Nearest edge: {edge.vertex1.position} - {edge.vertex2.position}")
        if self.current_player.can_build_road() and self.game_rules.is_valid_road_placement(edge, self.current_player):
            road = Road(vertex1=edge.vertex1, vertex2=edge.vertex2, player=self.current_player)
            edge.road = road
            self.current_player.roads -= 1
            print(f"Placed road at {edge.vertex1.position} - {edge.vertex2.position}")
        else:
            print("Invalid Road placement")
            
    def change_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        print(f"Changed to player {self.current_player_index}, {self.current_player.color}")
        