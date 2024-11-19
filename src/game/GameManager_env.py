from game.House import House
from game.City import City
from game.Road import Road
import numpy as np

'''
The gamemanger class oversees the progression of the game, managing player turns,
handling actions, updating the game state and enforcing game rules. It interacts closely
with the game board, players, and game rules classes to manage the game state.
It also has its own console to log game events and actions.

Args:
game_board (GameBoard): The game board containing all game elements, excluding player spesific information.
game_rules (GameRules): The game rules class that governs the rules of the game.
players (list): A list of player objects representing the players in the game.
console (Console): A console object to log game events and actions.

Attributes:
turn (int): The current turn number of the game.
max_turns (int): The maximum number of turns before the game ends.
game_board (GameBoard): Reference to the game board object.
game_rules (GameRules): Reference to the game rules object.
players (list): A list of players in the game.
console (Console): Reference to the console object.
current_player_index (int): The index of the current player in the players list.
current_player (Player): Reference to the current player object.
has_placed_piece (bool): Flag to indicate if the current player has placed a piece this turn.
player_passed_turn (bool): Flag to indicate if the current player has passed their turn.
dice_rolled (bool): Flag to indicate if the dice has been rolled this turn.
game_over (bool): Flag to indicate if the game is over.
gamestate (str): The current state of the game, either 'settle_phase' or 'normal_phase'.
starting_sub_phase (str): The current sub-phase of the starting phase, either 'house' or 'road'.
phase_transition (bool): Flag to indicate if the game is transitioning between phases.
game_ended_by_victory_points (bool): Flag to indicate if the game ended due to a player reaching 10 victory points.
starting_phase_players_stack (list): A list of players in the starting phase, used to determine turn order.
settlement_count (dict): A dictionary to keep track of the number of settlements placed by each player.
highlighted_vertecies (list): A list of highlighted vertices for valid house and city placements.
highlighted_edges (list): A list of highlighted edges for valid road placements.
last_placed_house_vertex (dict): A dictionary to keep track of the last placed house vertex for each player.
last_placed_road_edge (dict): A dictionary to keep track of the last placed road edge for each player.

Methods:
roll_phase(): Rolls the dice and collects resources for the current player.
robber(): Simple implementation of the robber taking resources from players with more than a given amount of resources.
is_turn_over(): Checks if the current player's turn is over.
pass_turn(): Passes the current player's turn.
check_if_game_ended(): Checks if the game has ended based on victory points or max turns.
change_player(): Changes the current player to the next player in the list.
roll_dice(): Rolls two dice and returns the sum of the two values.
trade_with_bank(trade_in_resource, get_back_resource): Trades resources with the bank at a 4:1 ratio.
check_tile_resources(roll): Checks the resources of the tiles based on the dice roll and collects resources for the players.
settlement_bonus(vertex): Gives the player resources based on the settlement bonus.
get_player_adj_resources_and_numbers(player, settlement_vertex=None): Gets the resource type and numbers of adjacent tiles for a player.
handle_action(action_type, action_params): Handles the action sent from the agent and updates the game state accordingly.
place_house(vertex): Places a house on the given vertex for the current player.
place_city(vertex): Places a city on the given vertex for the current player.
place_road(edge): Places a road on the given edge for the current player.
find_available_house__and_city_locations(): Finds the available vertices for house and city placements.
find_available_road_locations(): Finds the available edges for road placements.
remove_highlighted_locations(): Removes the highlighted vertices and edges.
simulate_settle_phase(): Simulates the settle phase of the game for the agents
simulate_place_house(vertex): method to place a house without needing player resources
simulate_place_road(edge): method to place a road without needing player resources

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
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
                
                resources_to_keep = total_resources // 2
                current_resources = dict(player.resources)
                
                sorted_resources = sorted(
                    current_resources.items(), 
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for resource in player.resources:
                    player.resources[resource] = 0
                    
                remainder_to_keep = resources_to_keep
                for resource, amount in sorted_resources:
                    if remainder_to_keep <= 0:
                        break
                    
                    keep_amount = min(amount, remainder_to_keep)
                    player.resources[resource] = keep_amount
                    remainder_to_keep -= keep_amount
                
                self.console.log(f"{player.get_color()} had to discard resources (kept {resources_to_keep})")
            else:
                self.console.log(f"{player.get_color()} did not have to discard resources, total resources: {total_resources}")
            
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
            self.console.log(f"Player {self.current_player.get_color()} won the game with {self.current_player.get_victory_points()}!")
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
                    if vertex.house:
                        player = vertex.house.player
                        if player.resources.get(tile.resource, 0) < 25:
                            player.add_resource(tile.resource, 1)
                            self.console.log(f"{player.get_color()} collected 1 {tile.resource}")
                        else:
                            print("Player has max resources")
                    elif vertex.city:
                        player = vertex.city.player
                        if player.resources.get(tile.resource, 0) < 24:
                            player.add_resource(tile.resource, 2)
                            self.console.log(f"{player.get_color()} collected 2 {tile.resource}")
                        elif player.resources.get(tile.resource, 0) == 24:
                            player.add_resource(tile.resource, 1)
                            self.console.log(f"{player.get_color()} collected 1 {tile.resource}")
                        else:
                            print("Player has max resources")
                        
    def settlement_bonus(self, vertex):
        adjacent_tiles = self.game_board.get_tiles_adj_to_vertex(vertex)
        for tile in adjacent_tiles:
            if tile.resource == 'desert':
                continue
            vertex.house.player.add_resource(tile.resource, 1)
            self.console.log(f"{vertex.house.player.get_color()} collected 1 {tile.resource} from settlement bonus")
                
    def get_player_adj_resources_and_numbers(self, player, settlement_vertex=None):
        resources = set()
        numbers = set()
        for vertex in self.game_board.vertices.values():
            if vertex.house and vertex.house.player == player and vertex != settlement_vertex:
                adj_tiles = self.game_board.get_tiles_adj_to_vertex(vertex)
                for tile in adj_tiles:
                    if tile.resource != 'desert':
                        resources.add(tile.resource)
                        numbers.add(tile.number)
        return resources, numbers
    
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
    
    # To simulate to make training more feasible and faster    
    def simulate_settle_phase(self):
        players = self.players
        game_board = self.game_board
        game_rules = self.game_rules
        console = self.console
        
        
        settle_phase_players_stack = players + players[::-1]
        player_resources = {player: set() for player in players}
        occupied_vertices = set()
        
        for player in settle_phase_players_stack:
            self.current_player_index = players.index(player)
            
            valid_vertecies = []
            for vertex in game_board.vertices.values():
                if vertex in occupied_vertices:
                    continue
                if not self.game_rules.is_valid_house_placement(vertex, player, phase='settle_phase'):
                    continue                
                resources = game_board.get_resources_adj_to_vertex(vertex)
                num_resources = len(resources)
                valid_vertecies.append((vertex, num_resources, resources))
                
            if not valid_vertecies:
                console.log(f"{player.get_color()} has no valid house placements")
                return False
            
            valid_vertecies.sort(key=lambda x: x[1], reverse=True)
            chosen_vertex, _, resources = valid_vertecies[0]
            occupied_vertices.add(chosen_vertex)
            player_resources[player].update(resources.keys())
            
            if not self.simulate_place_house(chosen_vertex):
                console.log(f"Could not place house for {player.get_color()}")
                return False
            
            valid_edges = []
            for neighbor in chosen_vertex.neighbors:
                edge_key = tuple(sorted([chosen_vertex.position, neighbor.position]))
                edge = game_board.edges.get(edge_key)
                if not edge:
                    continue
                if not game_rules.is_valid_road_placement(edge, player, phase='settle_phase', last_placed_house_vertex=chosen_vertex):
                    continue
                valid_edges.append(edge)
                
            if not valid_edges:
                console.log(f"{player.get_color()} has no valid road placements")
                return False
            
            chosen_edge = valid_edges[0]
            if not self.simulate_place_road(chosen_edge):
                console.log(f"Could not place road for {player.get_color()}")
                return False
        
        for player in players:
            unique_resources = {resource for resource in player_resources[player] if resource != 'desert'}
            if len(unique_resources) < 4:
                console.log(f"{player.get_color()} has less than 4 unique resources")
                return False
            
        return True
    
    def simulate_place_house(self, vertex):
        if self.game_rules.is_valid_house_placement(vertex, self.current_player, self.gamestate):
            house = House(vertex=vertex, player=self.current_player)
            vertex.house = house
            self.current_player.victory_points += 1
            self.current_player.settlements -= 1
            self.last_placed_house_vertex[self.current_player] = vertex
            
            # Add these settlement phase flags
            if self.gamestate == 'settle_phase':
                self.has_placed_piece = True
                self.starting_sub_phase = 'road'
                self.settlement_count[self.current_player] = self.settlement_count.get(self.current_player, 0) + 1
            
            return True
        return False
    
    def simulate_place_road(self, edge):
        if self.game_rules.is_valid_road_placement(edge, 
                                                   self.current_player, 
                                                   self.gamestate, 
                                                   self.last_placed_house_vertex.get(self.current_player)):
            road = Road(vertex1=edge.vertex1, vertex2=edge.vertex2, player=self.current_player)
            edge.road = road
            self.current_player.roads -= 1
            
             # Add these settlement phase flags
            if self.gamestate == 'settle_phase':
                self.has_placed_piece = True
                self.starting_sub_phase = 'house'
                self.player_passed_turn = True
                
            return True
        return False
            
        