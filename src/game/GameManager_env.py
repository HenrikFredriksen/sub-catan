from game.House import House
from game.City import City
from game.Road import Road
import numpy as np

class GameManager:
    def __init__(self, game_board, game_rules, players, console):
        self.turn = 0
        self.max_turns = 1000
        self.game_board = game_board
        self.game_rules = game_rules
        self.players = players
        self.console = console
        self.current_player_index = 0
        
        #FLAGS
        self.has_placed_piece = False
        self.player_passed_turn = False
        self.dice_rolled = False
        self.game_over = False
        self.gamestate = 'settle_phase'
        self.starting_sub_phase = 'house'
        self.phase_transition = False
        self.game_ended_by_victory_points = False
        
        self.starting_phase_players_stack = self.players + self.players[::-1]
        self.settlement_count = {player: 0 for player in self.players}
        
        self.highlighted_vertecies = []
        self.highlighted_edges = []
        self.last_placed_house_vertex = {}
        self.last_placed_road_edge = {}

        
    @property
    def current_player(self):
        return (self.starting_phase_players_stack[self.current_player_index] if self.turn == 0 
                else self.players[self.current_player_index])

    def roll_phase(self):
        if not self.dice_rolled:
            roll = self.roll_dice()
            self.dice_rolled = True
            if roll == 7:
                self.robber()
                self.console.log("Robber moves")
            else:
                self.check_tile_resources(roll)
                print(f"Resources collected for roll {roll}")
        else:
            self.console.log("Dice already rolled this turn")
            
    def robber(self):
        for player in self.players:
            total_resources = sum(player.resources.values())
            if total_resources > 7:
                resources = player.resources
                half = total_resources // 2
                # sort resources by amount
                sorted_resources = sorted(resources.items(), key=lambda x: x[1])
                
                for resource in player.resources:
                    player.resources[resource] = 0
                    
                resource_kept = 0
                for resource, amount in reversed(sorted_resources):
                    if resource_kept + amount <= half:
                        player.resources[resource] = amount
                        resource_kept += amount
                    else:
                        player.resources[resource] = half - resource_kept
                        break
                self.console.log(f"{player.get_color()} lost half of their resources")
            else:
                self.console.log(f"{player.get_color()} has {total_resources}, not robbed")
            
    def is_turn_over(self):
        if self.player_passed_turn:
            return True
        else:
            return False
        
    def pass_turn(self):
        if self.dice_rolled == False:
            self.roll_phase()
            
        if self.has_placed_piece:
            self.has_placed_piece = False
            
        self.player_passed_turn = True
        self.dice_rolled = False
        self.turn += 1
        self.console.log(f"{self.current_player.get_color()} passed their turn")

    def check_if_game_ended(self):
        if self.current_player.victory_points >= 10:
            self.console.log(f"Player {self.current_player.get_color()} won the game!")
            self.game_ended_by_victory_points = True
            self.game_over = True
            return True
        elif self.turn >= self.max_turns:
            self.console.log("Game over, max turns reached")
            self.game_over = True
            return True
        else:
            self.game_over = False
            return False
               
    def change_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        print(f"Changed to player {self.current_player_index}, {self.current_player.color}")
        
    def roll_dice(self):
        dice1 = np.random.randint(1, 7)
        dice2 = np.random.randint(1, 7)
        self.console.log(f"Dice rolled: {dice1}, {dice2}")
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
            self.console.log(f"{vertex.house.player.get_color()} collected 1 {tile.resource} from settlement bonus")
                
    def handle_action(self, action_type, action_params):
        print(f"Action: {action_type}, {action_params} sent from step")
        if action_type == 'place_house':
            self.place_house(action_params)
            if self.last_placed_house_vertex[self.current_player] != action_params:
                print(f"House was not placed at {action_params}")
                return False
            return True
        elif action_type == 'place_city':
            self.place_city(action_params)
            if self.last_placed_house_vertex[self.current_player] != action_params:
                print(f"City was not placed at {action_params}")
                return False
            return True
        elif action_type == 'place_road':
            self.place_road(action_params)
            if self.last_placed_road_edge[self.current_player] != action_params:
                print(f"Road was not placed at {action_params}")
                return False
            return True
        elif action_type == 'pass_turn':
            self.pass_turn()
            return True
        elif action_type == 'roll_dice':
            self.roll_phase()
            return True
        #implment trade with bank later maybe, not necessary for now
        elif action_type == 'trade_with_bank':
            self.trade_with_bank(action_params[0], action_params[1])
        else:
            print(f"Invalid action: {action_type}")
            return False

    def place_house(self, vertex):
        self.last_placed_house_vertex[self.current_player] = None
        if (self.current_player.can_build_settlement() and 
            self.game_rules.is_valid_house_placement(vertex, self.current_player, self.gamestate)):
            
            house = House(vertex=vertex, player=self.current_player)
            vertex.house = house
            
            self.current_player.victory_points += 1
            self.console.log(f"{self.current_player.get_color()} built a settlement +1VP, in total {self.current_player.victory_points}VP")
            self.current_player.settlements -= 1
            self.current_player.resources['wood'] -= 1
            self.current_player.resources['brick'] -= 1
            self.current_player.resources['sheep'] -= 1
            self.current_player.resources['wheat'] -= 1
            
            # update the highlighted locations
            #self.find_available_road_locations()
            #self.find_available_house__and_city_locations()
            
            self.last_placed_house_vertex[self.current_player] = vertex
            print(f"Settlement built at {vertex.position}")
            
            if self.gamestate == 'settle_phase':
                self.has_placed_piece = True
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
        self.last_placed_house_vertex[self.current_player] = None
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
            
            self.last_placed_house_vertex[self.current_player] = vertex
            
            self.console.log(f"{self.current_player.get_color()} built a city +1VP, in total {self.current_player.victory_points}VP")
            print(f"City built at {vertex.position}")
        else:
            print(f"Invalid City placement:\n" +
                  f"Player has resources? {self.current_player.can_build_city()}\n" +
                  f"City already placed? {vertex.city}")
            
    def place_road(self, edge):
        self.last_placed_road_edge[self.current_player] = None
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
            
            self.last_placed_road_edge[self.current_player] = edge
            
            self.console.log(f"Red placed road at {edge.vertex1.position} - {edge.vertex2.position}")
            
            if self.gamestate == 'settle_phase':
                self.has_placed_piece = True
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
        
    def find_available_road_locations(self):
        self.highlighted_edges = []
        for edge in self.game_board.edges.values():
            if (self.current_player.can_build_road() and 
                self.game_rules.is_valid_road_placement(edge, 
                                                       self.current_player, 
                                                       self.gamestate, 
                                                       self.last_placed_house_vertex.get(self.current_player))):
                self.highlighted_edges.append(edge)
                        
    def remove_highlighted_locations(self):
        self.highlighted_vertecies = []
        self.highlighted_edges = []
                        
                
        