from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
import random
import pickle
import os

from game.GameBoard import GameBoard
from game.GameManager_env import GameManager
from game.GameRules import GameRules
from game.Player import Player
from assets.PrintConsole import PrintConsole
from assets.Console import Console
from environment.CustomAgentSelector import CustomAgentSelector

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    
    env = CatanEnv(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
        
    env = wrappers.AssertOutOfBoundsWrapper(env)
    
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class CatanEnv(AECEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': 'catan_v0'}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            pygame.init()
            self.screen_width = 1400
            self.screen_height = 700
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Sub-Catan")
            self.clock = pygame.time.Clock()
            self.hex_size = 80
            self.number_size = 45
            self.load_resources()
            
            self.font_console = pygame.font.SysFont(None, 18)
            
            self.is_open = True
        else:
            self.tile_images = None
            self.number_images = None
            self.is_open = False
            
        self.agents = ['player_1', 'player_2', 'player_3', 'player_4']
        self.possible_agents = self.agents[:]
        self.starting_agents = self.agents + self.agents[::-1]
        self.agent_name_mapping = dict(zip(self.agents, (range(len(self.agents)))))        
        self._agent_selector = agent_selector(self.agents)
        
        self.game_board = GameBoard(self.tile_images, self.number_images)            
        self.game_board.set_screen_dimensions(1400, 700)
        self.game_rules = GameRules(self.game_board)
        self.players = [
            Player(player_id=0, color=(255, 0, 0), settlements=3, roads=13, cities=4, victory_points=2), # Red player
            Player(player_id=1, color=(0, 0, 255), settlements=3, roads=13, cities=4, victory_points=2), # Blue player
            Player(player_id=2, color=(20, 220, 20), settlements=3, roads=13, cities=4, victory_points=2), # Green player
            Player(player_id=3, color=(255, 165, 0), settlements=3, roads=13, cities=4, victory_points=2) # Orange player
        ]
        
        if render_mode == "human":
            self.console = Console(x=self.screen_width - 400, y=10, width=390, height=200, font=self.font_console)
        else:
            self.console = PrintConsole()
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_board.generate_board(board_radius=2)
        self.vertices_list = list(self.game_board.vertices.values())
        self.edges_list = list(self.game_board.edges.values())
        
        self.action_spaces = {
            agent: spaces.Discrete(self.calculate_action_space_size()) for agent in self.agents
        }
        
        enemy_state_size = (len(self.possible_agents) - 1) * 2 # Total resources and victory points for each enemy, two values per enemy
        
        obs_size = (
            self.calculate_board_state_size() + # Board state
            self.calculate_player_state_size() + # Resources, remaining pieces and victory points for the player
            enemy_state_size # Total resources and victory points for each enemy, two values per enemy
            #self.action_spaces[self.agents[0]].n # size of action mask
        )
        self.observation_spaces = {
            #agent: spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32) for agent in self.agents  
            agent: spaces.Dict(
                {
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.action_spaces[agent].n,), dtype=np.int8
                    ),
                    "observation": spaces.Box(
                        low=0, high=1, shape=(obs_size,), dtype=np.float32
                    ),
                }
            )   
            for agent in self.agents   
        }
                
        self.game_manager.gamestate = 'normal_phase'
        
        self.pass_action_index = 0
        self.roll_dice_action_index = 1
        self.step_count = 0
        self.was_placement_successful = False
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def calculate_action_space_size(self):
        num_vertices = len(self.vertices_list)
        num_edges = len(self.edges_list)
        # 1 for passing, 1 for dice throw, num_vertices for placing settlements and cities, num_edges for placing roads
        return 2 + num_vertices + num_vertices + num_edges 
    
    def calculate_board_state_size(self):
        num_vertices = len(self.game_board.vertices)
        num_edges = len(self.game_board.edges)
        num_tiles = len(self.game_board.tiles)
        # 2 for house and city, 1 for road, 6 for resources and 1 for number
        return num_vertices * 2 + num_edges * 1 + num_tiles * 6 + num_tiles * 1
    
    def calculate_player_state_size(self):
        num_resources = 5
        num_different_pieces = 3
        victory_points = 1
        return num_resources + num_different_pieces + victory_points
    
    def reset(self, seed=None, return_info=False, options=False):
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # reset counter and flags
        self.step_count = 0
        
        # reset agents and selector
        self.agents = ['player_1', 'player_2', 'player_3', 'player_4']
        self.possible_agents = self.agents[:]
        self.starting_agents = self.agents + self.agents[::-1]
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.agents[0]
        
        # reset game state dictionaries
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {'seed': seed} for agent in self.agents}
        
        self.console.log(f"Starting new game with agents: {self.agents}")
        self.console.log(f"Terminations: {self.terminations}")

        self.agent_name_mapping = dict(zip(self.agents, (range(len(self.agents)))))

        # reset game components
        self.game_board = self.load_random_board_normal_phase()
        self.game_board.set_screen_dimensions(1400, 700)
        self.game_rules = GameRules(self.game_board)
        self.players = [
            Player(player_id=0, color=(255, 0, 0), settlements=3, roads=13, cities=4, victory_points=2), # Red player
            Player(player_id=1, color=(0, 0, 255), settlements=5, roads=13, cities=4, victory_points=2), # Blue player
            Player(player_id=2, color=(20, 220, 20), settlements=3, roads=13, cities=4, victory_points=2), # Green player
            Player(player_id=3, color=(255, 165, 0), settlements=3, roads=13, cities=4, victory_points=2) # Orange player
        ]
        
        player_id_map = {player.player_id: player for player in self.players}
        
        for vertex in self.game_board.vertices.values():
            if vertex.house:
                vertex_house_player_id = vertex.house.player.player_id
                vertex.house.player = player_id_map[vertex_house_player_id]
            if vertex.city:
                vertex_city_player_id = vertex.city.player.player_id
                vertex.city.player = player_id_map[vertex_city_player_id]
        for edge in self.game_board.edges.values():
            if edge.road:
                edge_road_player_id = edge.road.player.player_id
                edge.road.player = player_id_map[edge_road_player_id]
                
        # reset game manager and generate new board
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_manager.gamestate = 'normal_phase'
        self.game_manager.dice_rolled = False
        #self.game_board.generate_board(board_radius=2)
        
        print(f"Players in player_id_map: {list(player_id_map.keys())}")
        houses_with_settle_bonus = {player.player_id: False for player in self.players}
        for vertex in self.game_board.vertices.values():
            if vertex.house:
                print(f"House player_id: {vertex.house.player.player_id}")
                player = vertex.house.player
                if not houses_with_settle_bonus[player.player_id]:
                    self.game_manager.settlement_bonus(vertex)
                    houses_with_settle_bonus[player.player_id] = True
                
        
        # reset vertices and edges lists
        self.vertices_list = list(self.game_board.vertices.values())
        self.edges_list = list(self.game_board.edges.values())

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.load_resources()
            self.game_board.tile_images = self.tile_images
            self.game_board.number_images = self.number_images
            self.render()
                 
        obs = self.observe(self.agent_selection)
        
        if return_info:
            return obs, self.infos[self.agent_selection]
        return obs
    
    def load_random_board_normal_phase(self):
        board_files = os.listdir('normal_phase_boards')
        if not board_files:
            raise Exception("No saved boards found in 'saved_boards' directory")
        random_board_file = random.choice(board_files)
        file_path = os.path.join('normal_phase_boards', random_board_file)
        with open(file_path, 'rb') as f:
            game_board = pickle.load(f)
            print(f"Loaded board from {file_path}")
        return game_board
    
    def observe(self, agent):
        board_state = self.get_board_state()
        player_state = self.get_player_state(agent)
        enemy_state = self.get_enemy_state(agent).flatten()

        action_mask = self.get_action_mask(agent)
        observation = np.concatenate([board_state, player_state, enemy_state])
        
        
        # takes a dictionary of observations for each agent
        observation_dict = {
            "action_mask": action_mask,
            "observation": observation
        }
        # Debugging statements
        #print(f"Agent: {agent}")
        #print(f"Board state size: {board_state.size}")
        #print(f"Player state size: {player_state.size}")
        #print(f"Enemy state size: {enemy_state.size}")
        #print(f"Total observation size: {observation.size}, Expected: {self.observation_spaces[agent]['observation'].shape[0]}")
        #print(f"Total action mask size: {action_mask.size}, Expected: {self.observation_spaces[agent]['action_mask'].shape[0]}")
        
        return observation_dict
    
    def get_action_mask(self, agent):
        valid_actions = self.get_valid_actions(agent)
        action_mask = np.zeros(self.calculate_action_space_size(), dtype=np.int8)
        action_mask[valid_actions] = 1.0
        return action_mask
    
    def get_board_state(self):
        
        vertex_states = []
        for vertex in self.game_board.vertices.values():
            if vertex.house:
                vertex_state = np.array([1, 0], dtype=np.float32)
            elif vertex.city:
                vertex_state = np.array([0, 1], dtype=np.float32)
            else:
                vertex_state = np.array([0, 0], dtype=np.float32)
            vertex_states.append(vertex_state)
        # vertex_states should be number of vertices x 2
        vertex_states = np.array(vertex_states).flatten()
                
        edge_states = []
        for edge in self.game_board.edges.values():
            if edge.road:
                edge_state = np.array([1], dtype=np.float32)
            else:
                edge_state = np.array([0], dtype=np.float32)
            edge_states.append(edge_state)
        # edge_states should be number of edges x 1
        edge_states = np.array(edge_states).flatten()
        
        tile_states = []
        resource_to_idx = {'wood': 0, 'brick': 1, 'sheep': 2, 'wheat': 3, 'ore': 4, 'desert': 5}
        for tile in self.game_board.tiles.values():
            resource_one_hot = np.zeros(7, dtype=np.float32)
            tile_number = tile.number / 12
            resource_one_hot[6] = tile_number
            resource_idx = resource_to_idx.get(tile.resource, -1)
            if resource_idx >= 0:
                resource_one_hot[resource_idx] = 1
            tile_states.append(resource_one_hot)
        # tile_states should be number of tiles x 7, 6 for resources and 1 for number
        tile_states = np.array(tile_states).flatten()
        
        # board_state should be number of vertices x 2 + number of edges x 1 + number of tiles x 7
        board_state = np.concatenate([vertex_states, edge_states, tile_states])
        
        return board_state
    
    def get_player_state(self, agent):
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        
        resources = np.array([
            player.resources['wood'],
            player.resources['brick'],
            player.resources['sheep'],
            player.resources['wheat'],
            player.resources['ore']
        ], dtype=np.float32)
        resources /= 25
        
        remaining_pieces = np.array([
            player.settlements / 5,
            player.roads / 15,
            player.cities / 4
        ], dtype=np.float32)
        
        victory_points = np.array([player.victory_points / 10], dtype=np.float32)
        # player_state should be 5 resources + 3 remaining pieces + 1 victory point
        player_state = np.concatenate([resources, remaining_pieces, victory_points])
        return player_state
    
    def get_enemy_state(self, agent):
        enemy_states = []
        for enemy in self.possible_agents:
            player_idx = self.agent_name_mapping[enemy]
            player = self.players[player_idx]
            if enemy != agent:
                enemy_total_resources = sum(player.resources.values()) / (25 * 5)
                enemy_vp = player.victory_points / 10
                enemy_state = np.array([enemy_total_resources, enemy_vp], dtype=np.float32)
                enemy_states.append(enemy_state)
                
        return np.array(enemy_states, dtype=np.float32)
    
    def get_victory_points(self):
        victory_points =  {}
        for player in self.players:
            agent_id = 'player_' + str(self.players.index(player) + 1)
            victory_points[agent_id] = player.victory_points
        return victory_points
        
    def step(self, action):
        self.step_count += 1
        self.console.log(f"step_{self.step_count}, agent: {self.agent_selection}, gamestate: {self.game_manager.gamestate}")
        
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        self.game_manager.current_player_index = player_idx
        
        if self.terminations[agent] or self.truncations[agent]:
            self.rewards[agent] = 0.0
            self._was_dead_step(action)
            return
        
        valid_actions = self.get_valid_actions(agent)
        self.console.log(f"step_Valid actions: {valid_actions}")
        if action not in valid_actions:
            # Action is invalid, agent terminated
            self.rewards[agent] = -2.0
            self.terminations[agent] = True
            self.console.log(f"step_Invalid action: {action}, terminated.")
            self._was_dead_step(action)
            return
        else:
            action_type, action_param = self.decode_action(action)
            self.was_placement_successful = self.game_manager.handle_action(action_type, action_param)

            self.rewards[agent] = self.calculate_reward(agent, action_type)
            self.console.log(f"Agent {agent} executed action {action_type}, reward: {self.rewards[agent]}")
        
        game_over = self.game_manager.check_if_game_ended()
        if game_over:
            for ag in self.agents:
                self.terminations[ag] = True
                self.rewards[ag] += 20 if self.game_manager.game_ended_by_victory_points else 0
            self.infos[agent]['reason'] = 'Victory'
            self.console.log(f"step_Game Over!")
            self.console.log(f"step_Terminating all agents: {self.terminations}")

        # Settle phase
        if self.game_manager.gamestate == 'settle_phase':
            # If the current sub-phase is road, it means a house was placed this step, so the next agent should be the same
            if self.game_manager.starting_sub_phase == 'road' and self.game_manager.has_placed_piece:
                self.agent_selection = agent
            # If the current sub-phase is house, it means a road was placed this step, so the next agent should be the next one
            elif self.game_manager.starting_sub_phase == 'house' and self.game_manager.has_placed_piece:
                self.agent_selection = self._agent_selector.next()
                  
            if self._agent_selector.is_last():
                self._agent_selector = agent_selector(self.agents)
                self.agent_selection = self.agents[0]
                self.game_manager.gamestate = 'normal_phase'
                self.console.log(f"step_Phase transition to normal phase")
        # Normal phase
        else:
            # If the turn is over, the next agent should be the next one
            if self.game_manager.is_turn_over():
                self.agent_selection = self._agent_selector.next()
            # If the agent is not done doing their actions, they should continue
            else:
                self.agent_selection = agent
        
        try:
            self._accumulate_rewards()
            
            if self.render_mode == "human":
                self.render()
        except TypeError as e:
            print(f"TypeError encountered: {e}")
            print(f"self.rewards[{agent}] = {self.rewards[agent]}")
            print(f"self._cumulative_rewards[{agent}] = {self._cumulative_rewards[agent]}")
            self.rewards[agent] = 0.0
            self._accumulate_rewards()
            
    def decode_action(self, action):
        print(f"Decoding action: {action}")
        num_vertices = len(self.vertices_list)
        
        if action == self.pass_action_index:
            return 'pass_turn', None
        elif action == self.roll_dice_action_index:
            return 'roll_dice', None
        elif 2 <= action < num_vertices:
            vertex_index = action - 2
            vertex = self.vertices_list[vertex_index]
            return 'place_house', vertex
        elif num_vertices <= action < 2 * num_vertices:
            vertex_index = action - num_vertices - 2
            vertex = self.vertices_list[vertex_index]
            return 'place_city', vertex
        else:
            edge_index = action - 2 * num_vertices - 2
            edge = self.edges_list[edge_index]
            return 'place_road', edge
    
    def get_valid_actions(self, agent):
        if self.game_manager.gamestate == 'settle_phase':
            valid_actions = self.get_settle_phase_actions(agent)
        elif self.game_manager.gamestate == 'normal_phase':
            valid_actions = self.get_normal_phase_actions(agent)
        else:
            valid_actions = []
        return valid_actions
    
    def get_settle_phase_actions(self, agent):
        valid_actions = [] 
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        self.game_manager.current_player_index = player_idx
        
        self.game_manager.find_available_house__and_city_locations()
        self.game_manager.find_available_road_locations()
        
        num_vertices = len(self.vertices_list)
        
        if self.game_manager.starting_sub_phase == 'house':
            for vertex in self.game_manager.highlighted_vertecies:
                idx = self.vertices_list.index(vertex)
                action = 2 + idx # 2 for pass and roll dice actions
                valid_actions.append(action)
            
        elif self.game_manager.starting_sub_phase == 'road':    
            for edge in self.game_manager.highlighted_edges:
                idx = self.edges_list.index(edge)
                # 2 for pass and roll dice actions, 2 * num_vertices for placing settlements
                action = 2 + 2 * num_vertices + idx 
                valid_actions.append(action)
                
        return valid_actions
    
    def get_normal_phase_actions(self, agent):
        valid_actions = [self.pass_action_index]
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        self.game_manager.current_player_index = player_idx
        
        if not self.game_manager.dice_rolled:
            valid_actions.append(self.roll_dice_action_index)
            return valid_actions
        else:
            self.game_manager.find_available_house__and_city_locations()
            self.game_manager.find_available_road_locations()
            
            num_vertices = len(self.vertices_list)
            
            for vertex in self.game_manager.highlighted_vertecies:
                idx = self.vertices_list.index(vertex)
                if vertex.house is None and vertex.city is None:
                    action = 2 + idx
                    valid_actions.append(action)
                elif vertex.house is not None and vertex.house.player == player and vertex.city is None:
                    action = 2 + num_vertices + idx
                    valid_actions.append(action)
                    
            for edge in self.game_manager.highlighted_edges:
                idx = self.edges_list.index(edge)
                action = 2 + 2 * num_vertices + idx
                valid_actions.append(action)
                
            return valid_actions
    
    def calculate_reward(self, agent, action_type):
        reward = 0
        if action_type == 'roll_dice':
            return 1.0
        if self.was_placement_successful:
            if action_type == 'place_house':
                vertex = self.game_manager.last_placed_house_vertex.get(self.agent_selection)
                if vertex:
                    adj_tiles = self.game_board.get_tiles_adj_to_vertex(vertex)
                    reward += len(adj_tiles) * 2

                    resources = self.game_board.get_resources_adj_to_vertex(vertex)
                    unique_resources = set(resources.keys()) - {'desert'}
                    reward += len(unique_resources) * 3
                return reward
            elif action_type == 'place_city':
                return 6
            elif action_type == 'place_road':
                return 0.0
            elif action_type == 'pass_turn':
                return -1.1
        else:
            return -1.0
    
    def render(self):
        if not self.is_open:
            return None
        
        self.screen.fill((100, 140, 250))
        self.game_board.draw(self.screen)
        self.draw_game_state()
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_open = False
                    pygame.quit()
                    
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), 
            axes=(1, 0, 2)
        )
                
    def draw_game_state(self):
        y_offset = 10
        for player in self.game_manager.players:
            text = player.__str__()
            text_font = pygame.font.Font(None, 20)
            text_surface = text_font.render(text, True, player.color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20

        
    def load_resources(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            base_path = os.path.join(os.path.dirname(__file__), '../../img/')
            self.tile_width = int(self.hex_size * np.sqrt(3))
            self.tile_height = int(self.hex_size * 2)
            # Load tile images
            self.tile_images = {
                "wood": pygame.transform.scale(
                    pygame.image.load(os.path.join(base_path, 'terrainHexes/', "forest.png")),
                    (self.tile_width * 1.02, self.tile_height * 1.02)
                ),
                "brick": pygame.transform.scale(
                    pygame.image.load(os.path.join(base_path, 'terrainHexes/', "hills.png")),
                    (self.tile_width * 1.02, self.tile_height * 1.02)
                ),
                "sheep": pygame.transform.scale(
                    pygame.image.load(os.path.join(base_path, 'terrainHexes/', "pasture.png")),
                    (self.tile_width * 1.02, self.tile_height * 1.02)
                ),
                "wheat": pygame.transform.scale(
                    pygame.image.load(os.path.join(base_path, 'terrainHexes/', "field.png")),
                    (self.tile_width * 1.02, self.tile_height * 1.02)
                ),
                "ore": pygame.transform.scale(
                    pygame.image.load(os.path.join(base_path, 'terrainHexes/', "mountain.png")),
                    (self.tile_width * 1.02, self.tile_height * 1.02)
                ),
                "desert": pygame.transform.scale(
                    pygame.image.load(os.path.join(base_path, 'terrainHexes/', "desert.png")),
                    (self.tile_width * 1.02, self.tile_height * 1.02)
                )
            }
            # Load number images
            self.number_images = {}
            for number in range(2, 13):
                if number != 7:
                    image_path = os.path.join(base_path, 'numbers/', f"{number}.png")
                    self.number_images[number] = pygame.transform.scale(
                        pygame.image.load(image_path),
                        (self.number_size, self.number_size)
                    )
        else:
            self.tile_images = None
            self.number_images = None

    def close(self):
        if (self.render_mode == "human" and self.is_open) or (self.render_mode == "rgb_array" and self.is_open):
            pygame.quit()
            self.is_open = False
        
        