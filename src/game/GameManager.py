from game.House import House
from game.Road import Road
import numpy as np

class GameManager:
    def __init__(self, game_board, game_rules, players):
        self.turn = 0
        self.game_board = game_board
        self.game_rules = game_rules
        self.players = players
        self.current_player_index = 0
        self.game_over = False
        
    @property
    def current_player(self):
        return self.players[self.current_player_index]
    
    def update(self):
        if self.game_over:
            return
        
        self.turn += 1
        print(f"Turn {self.turn}")
        print(f"Current player: {self.current_player_index}, {self.current_player.color}")
        self.current_player_index = self.turn % len(self.players)
        print(f"Next player: {self.current_player_index}, {self.current_player.color}")
        
        for player in self.players:
            if player.victory_points >= 10:
                print(f"Player {player} wins!")
                self.game_over = True
        else:
            roll = self.roll_dice()
            if roll == 7:
                #implement robber
                print("Robber moves")
                
            else:
                self.check_tile_resources(roll)
                print(f"Resources collected for roll {roll}")
            self.change_player()
            
    def check_tile_resources(self, roll):
        for tile in self.game_board.tiles.values():
            if tile.get_number() == roll:
                for vertex in self.game_board.get_tile_vertices(tile):
                    # check if there is a house on the vertex
                    if vertex.house:
                        vertex.house.player.add_resource(tile.resource, 1)
    
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
            self.current_player.victory_points += 1
            self.current_player.settlements -= 1
            self.current_player.resources['wood'] -= 1
            self.current_player.resources['brick'] -= 1
            self.current_player.resources['sheep'] -= 1
            self.current_player.resources['wheat'] -= 1
            print(f"Placed house at {vertex.position}")
        else:
            print("Invalid House placement")
            
    def place_road(self, edge):
        print(f"Nearest edge: {edge.vertex1.position} - {edge.vertex2.position}")
        if self.current_player.can_build_road() and self.game_rules.is_valid_road_placement(edge, self.current_player):
            road = Road(vertex1=edge.vertex1, vertex2=edge.vertex2, player=self.current_player)
            edge.road = road
            self.current_player.roads -= 1
            self.current_player.resources['wood'] -= 1
            self.current_player.resources['brick'] -= 1
            print(f"Placed road at {edge.vertex1.position} - {edge.vertex2.position}")
        else:
            print("Invalid Road placement")
            
    def change_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        print(f"Changed to player {self.current_player_index}, {self.current_player.color}")
        
    def roll_dice(self):
        dice1 = np.random.randint(1, 7)
        dice2 = np.random.randint(1, 7)
        print(f"Dice rolled: {dice1}, {dice2}")
        return dice1 + dice2
        