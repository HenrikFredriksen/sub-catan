from game.House import House
from game.City import City
from game.Road import Road
import numpy as np

class GameManager:
    def __init__(self, game_board, game_rules, players, console):
        self.turn = 0
        self.max_turns = 400
        self.game_board = game_board
        self.game_rules = game_rules
        self.players = players
        self.console = console
        self.current_player_index = 0
        self.player_passed_turn = False
        self.dice_rolled = False
        self.game_over = False
        self.highlighted_vertecies = []
        self.highlighted_edges = []
        self.gamestate = 'settle_phase'
        self.starting_sub_phase = 'house'
        self.starting_phase_players_stack = self.players + self.players[::-1]
        self.settlement_count = {player: 0 for player in self.players}
        self.last_placed_house_vertex = {}

        
    @property
    def current_player(self):
        return (self.starting_phase_players_stack[self.current_player_index] if self.turn == 0 
                else self.players[self.current_player_index])
    
    def start_game(self):
        self.update()
        
    def update(self):
        if self.gamestate == 'settle_phase':
            self.console.log("Starting phase")
            self.starting_phase()
        elif self.gamestate == 'normal_phase':
            self.normal_phase()
            self.find_available_house__and_city_locations()
            self.find_available_road_locations()
        elif self.gamestate == 'player_won':
            self.player_won_phase()
            
    def normal_phase(self):
        for player in self.players:
            if player.victory_points >= 10:
                self.gamestate = 'player_won'
                self.player_won_phase()
        if self.game_over:
            self.console.log("Game over")
            return
        
        self.roll_phase()
            
        if self.turn >= 1:
            self.change_player()         
        self.turn += 1
        
    def roll_phase(self):
        if not self.dice_rolled:
            roll = self.roll_dice()
            if roll == 7:
                #implement robber
                self.console.log("Robber moves")
            else:
                self.check_tile_resources(roll)
                print(f"Resources collected for roll {roll}")
        else:
            self.console.log("Dice already rolled this turn")
            
    def is_turn_over(self):
        if self.player_passed_turn:
            return True
        else:
            return False
        
        
    def pass_turn(self):
        self.player_passed_turn = True
        self.dice_rolled = False
        self.console.log(f"{self.current_player.get_color()} passed their turn")
        
        
    def starting_phase(self):
        if self.starting_sub_phase == 'house':
            self.find_available_house__and_city_locations()
            self.console.log(f"{self.current_player.get_color()}'s turn, place a house")
        elif self.starting_sub_phase == 'road':
            self.find_available_road_locations()
            self.console.log(f"{self.current_player.get_color()}'s turn, place a road")

    def player_won_phase(self):
        self.console.log(f"Player {self.current_player.get_color()} won the game!")
        self.game_over = True
               
    def change_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        print(f"Changed to player {self.current_player_index}, {self.current_player.color}")
        
    def roll_dice(self):
        dice1 = np.random.randint(1, 7)
        dice2 = np.random.randint(1, 7)
        self.console.log(f"Dice rolled: {dice1}, {dice2}")
        print(f"Dice rolled: {dice1}, {dice2}")
        return dice1 + dice2
    
    def trade_with_bank(self, trade_in_resource, get_back_resource):
        if self.current_player.can_trade_with_bank(trade_in_resource):
            self.current_player.resources[trade_in_resource] -= 4
            self.current_player.resources[get_back_resource] += 1
            self.console.log(f"{self.current_player.get_color()} traded 4 {trade_in_resource} for 1 {get_back_resource}")
        else:
            self.console.log(f"{self.current_player.get_color()} does not have enough resources to trade")

    def check_tile_resources(self, roll):
        for tile in self.game_board.tiles.values():
            if tile.get_number() == roll:
                for vertex in self.game_board.get_tile_vertices(tile):
                    # check if there is a house on the vertex
                    if vertex.house:
                        # check if the player has max resources
                        if vertex.house.player.resources.get(tile.resource) < 25:
                            self.console.log(f"{vertex.house.player.get_color()} collected 1 {tile.resource}")
                            vertex.house.player.add_resource(tile.resource, 1)
                        else:
                            print("Player has max resources")
                    elif vertex.city:
                        if vertex.city.player.resources.get(tile.resource) < 24:
                            self.console.log(f"{vertex.city.player.get_color()} collected 2 {tile.resource}")
                            vertex.city.player.add_resource(tile.resource, 2)
                        elif vertex.city.player.resources.get(tile.resource) == 24:
                            self.console.log(f"{vertex.city.player.get_color()} collected 1 {tile.resource}")
                            vertex.city.player.add_resource(tile.resource, 1)
                        else:
                            print("Player has max resources")
                        
    def settlement_bonus(self, vertex):
        adjacent_tiles = self.game_board.get_tiles_adj_to_vertex(vertex)
        for tile in adjacent_tiles:
            if tile.resource == 'desert':
                continue
            vertex.house.player.add_resource(tile.resource, 1)
    
    def handle_click(self, mouse_pos):
        print(f"Mouse clicked at: {mouse_pos}")
        proximity_radius = self.game_board.hex_size / 4
        
        nearest_vertex = self.game_board.find_nearest_vertex(mouse_pos, proximity_radius)
        nearest_edge = self.game_board.find_nearest_edge(mouse_pos, proximity_radius)
        
        if self.gamestate == 'settle_phase':
            if self.starting_sub_phase == 'house' and nearest_vertex:
                self.place_house(nearest_vertex)
                self.console.log(f"{self.current_player.get_color()} placed a house")
                self.starting_sub_phase = 'road'
                self.remove_highlighted_locations()
                self.starting_phase()
            elif self.starting_sub_phase == 'road' and nearest_edge:
                self.place_road(nearest_edge)
                self.console.log(f"{self.current_player.get_color()} placed a road")
                self.starting_sub_phase = 'house'
                self.remove_highlighted_locations()
                self.current_player_index += 1
                
                if self.current_player_index < len(self.starting_phase_players_stack):
                    self.starting_phase()
                    
                if self.current_player_index >= len(self.starting_phase_players_stack):
                    self.current_player_index = 0
                    self.gamestate = 'normal_phase'
                    self.last_placed_house_vertex = {}
                    self.update()
        
        else:
            if nearest_vertex and nearest_edge:
                vertex_distance = np.hypot(mouse_pos[0] - nearest_vertex.position[0], 
                                           mouse_pos[1] - nearest_vertex.position[1])
                edge_distance = np.hypot(
                    mouse_pos[0] - (nearest_edge.vertex1.position[0] + nearest_edge.vertex2.position[0]) // 2, 
                    mouse_pos[1] - (nearest_edge.vertex1.position[1] + nearest_edge.vertex2.position[1]) // 2)
                
                if vertex_distance < edge_distance:
                    if nearest_vertex.house:
                        self.place_city(nearest_vertex)
                    else:
                        self.place_house(nearest_vertex)
                else:
                    self.place_road(nearest_edge)
                    
            elif nearest_vertex:
                if nearest_vertex.house:
                    self.place_city(nearest_vertex)
                else:
                    self.place_house(nearest_vertex)
            elif nearest_edge:
                self.place_road(nearest_edge)
            else:
                print("No vertex or edge found")
                
    def handle_action(self, action_type, action_params):
        if action_type == 'place_house':
            self.place_house(action_params)
        elif action_type == 'place_city':
            self.place_city(action_params)
        elif action_type == 'place_road':
            self.place_road(action_params)
        elif action_type == 'pass_turn':
            self.pass_turn()
        #implment trade with bank later maybe, not necessary for now
        elif action_type == 'trade_with_bank':
            self.trade_with_bank(action_params[0], action_params[1])
        else:
            print(f"Invalid action: {action_type}")
            
    def end_turn(self):
        #not implemented yet
        #if self.gamestate == 'settle_phase':
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        

    def place_house(self, vertex):
        print(f"Place at vertex: {vertex.position}")
        if (self.current_player.can_build_settlement() and 
            self.game_rules.is_valid_house_placement(vertex, self.current_player, self.gamestate)):
            
            house = House(vertex=vertex, player=self.current_player)
            vertex.house = house
            
            
            self.current_player.victory_points += 1
            self.console.log(f"{self.current_player.get_color()} built a house +1VP")
            self.current_player.settlements -= 1
            self.current_player.resources['wood'] -= 1
            self.current_player.resources['brick'] -= 1
            self.current_player.resources['sheep'] -= 1
            self.current_player.resources['wheat'] -= 1
            
            # update the highlighted locations
            #self.find_available_road_locations()
            #self.find_available_house__and_city_locations()
            
            print(f"Placed house at {vertex.position}")
            
            if self.gamestate == 'settle_phase':
                self.last_placed_house_vertex[self.current_player] = vertex
                self.starting_sub_phase = 'road'
                
                # check if house is the second one placed, if so, give settlement bonus
                self.settlement_count[self.current_player] += 1
                if self.settlement_count[self.current_player] == 2:
                    self.game_rules.starting_settlement_bonus(vertex)
                    self.settlement_bonus(vertex)
                
                self.player_passed_turn = True
        else:
            print(f"Invalid House placement:\n" +
                  f"Player has resources? {self.current_player.can_build_settlement()}\n" + 
                  f"House already placed? {vertex.house}")

    def place_city(self, vertex):
        print(f"Place at vertex: {vertex.position}")
        if (self.current_player.can_build_city() 
            and self.game_rules.is_valid_city_placement(vertex, self.current_player)
            ):
            city = City(vertex=vertex, player=self.current_player)
            vertex.city = city
            
            vertex.house = None
            self.current_player.cities -= 1
            self.current_player.settlements += 1
            self.current_player.resources['wheat'] -= 2
            self.current_player.resources['ore'] -= 3
            self.current_player.victory_points += 1
            
            # update the highlighted locations
            #self.find_available_road_locations()
            #self.find_available_house__and_city_locations()
            
            self.console.log(f"{self.current_player.get_color()} built a city +1VP")
            print(f"Placed city at {vertex.position}")
        else:
            print(f"Invalid City placement:\n" +
                  f"Player has resources? {self.current_player.can_build_city()}\n" +
                  f"City already placed? {vertex.city}")
            
    def place_road(self, edge):
        print(f"Place at edge: {edge.vertex1.position} - {edge.vertex2.position}")
        if (self.current_player.can_build_road() 
            and self.game_rules.is_valid_road_placement(edge, 
                                                        self.current_player, 
                                                        self.gamestate, 
                                                        self.last_placed_house_vertex.get(self.current_player))
            ):
            road = Road(vertex1=edge.vertex1, vertex2=edge.vertex2, player=self.current_player)
            edge.road = road
            
            self.current_player.roads -= 1
            self.current_player.resources['wood'] -= 1
            self.current_player.resources['brick'] -= 1
            
            # update the highlighted locations
            #self.find_available_road_locations()
            #self.find_available_house__and_city_locations()
            
            self.console.log(f"{self.current_player.get_color()} built a road")
            print(f"Placed road at {edge.vertex1.position} - {edge.vertex2.position}")
            
            if self.gamestate == 'settle_phase':
                self.player_passed_turn = True
                self.starting_sub_phase = 'house'
        else:
            print(f"Invalid Road placement:\n" +
                  f"Player has resources? {self.current_player.can_build_road()}\n" +
                  f"nRoad already placed? {edge.road}")
        
    def find_available_house__and_city_locations(self):
        self.highlighted_vertecies = []
        for vertex in self.game_board.vertices.values():
            if self.gamestate == 'settle_phase':
                if (self.current_player.can_build_settlement() and 
                    self.game_rules.is_valid_house_placement(vertex, 
                                                             self.current_player, 
                                                             self.gamestate)):
                    self.highlighted_vertecies.append(vertex)
            #this could be optimized further
            elif self.gamestate == 'normal_phase':
                # house placement
                if (self.current_player.can_build_settlement() and 
                    self.game_rules.is_valid_house_placement(vertex, 
                                                             self.current_player, 
                                                             self.gamestate)):
                    self.highlighted_vertecies.append(vertex)
                # city placement
                elif (self.current_player.can_build_city() and 
                      self.game_rules.is_valid_city_placement(vertex, self.current_player)):
                    self.highlighted_vertecies.append(vertex)
        print(f"Available house locations: {len(self.highlighted_vertecies)}")
        
    def find_available_road_locations(self):
        self.highlighted_edges = []
        for edge in self.game_board.edges.values():
            if (self.current_player.can_build_road() and 
                self.game_rules.is_valid_road_placement(edge, 
                                                       self.current_player, 
                                                       self.gamestate, 
                                                       self.last_placed_house_vertex.get(self.current_player))):
                self.highlighted_edges.append(edge)
        print(f"Available road locations: {len(self.highlighted_edges)}")
        
    def remove_highlighted_locations(self):
        self.highlighted_vertecies = []
        self.highlighted_edges = []
                        
                
        