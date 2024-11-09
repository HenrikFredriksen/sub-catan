from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

from game.GameBoard import GameBoard
from game.GameManager import GameManager
from game.GameRules import GameRules
from game.Player import Player
from assets.Console import Console
from assets.PrintConsole import PrintConsole

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    
    env = CatanEnv(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
        
    env = wrappers.AssertOutOfBoundsWrapper(env)
    
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class CatanEnv(AECEnv):
    metadata = {'render.modes': ['human'], 'name': 'catan_v0'}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.agents = ['player1', 'player2', 'player3', 'player4']
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, (range(len(self.agents)))))
        self._agent_selector = agent_selector(self.agents)
        
        self.game_board = GameBoard()
        self.game_rules = GameRules(self.game_board)
        self.players = [
            Player(color=(255, 0, 0), settlements=5, roads=15, cities=4), # Red player
            Player(color=(0, 0, 255), settlements=5, roads=15, cities=4), # Blue player
            Player(color=(20, 220, 20), settlements=5, roads=15, cities=4), # Green player
            Player(color=(255, 165, 0), settlements=5, roads=15, cities=4) # Orange player
        ]
        
        if render_mode == "human":
            pygame.init()
            self.console_window = pygame.display.set_mode((300, 400))
            pygame.display.set_caption("Catan Console")
            font = pygame.font.SysFont(None, 18)
            self.console = Console(x=0, y=0, width=300, height=400, font=font)
        else:
            self.console = PrintConsole()
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_board.generate_board(board_radius=2)
        
        self.action_spaces = {agent: spaces.Discrete(self.calculate_action_space_size()) for agent in self.agents}
        
        enemy_state_size = (len(self.agents) - 1) * 2 # Total resources and victory points for each enemy, two values per enemy
        self.observation_spaces = {agent: spaces.Dict({
            'board_state': spaces.Box(low=0, high=1, shape=(self.calculate_board_state_size(),), dtype=np.float32), 
            'player_state': spaces.Box(low=0, high=1, shape=(self.calculate_player_state_size(),), dtype=np.float32),
            'enemy_state': spaces.Box(low=0, high=1, shape=(enemy_state_size,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.action_spaces[agent].n,), dtype=np.float32)
        }) for agent in self.agents}
        
        self.render_mode = render_mode
        
    def calculate_action_space_size(self):
        num_vertices = len(self.game_board.vertices)
        num_edges = len(self.game_board.edges)
        return num_vertices + num_edges
    
    def calculate_board_state_size(self):
        num_vertices = len(self.game_board.vertices)
        num_edges = len(self.game_board.edges)
        num_tiles = len(self.game_board.tiles)
        
        return num_vertices * 2 + num_edges * 1 + num_tiles * 5
    
    def calculate_player_state_size(self):
        num_resources = 5
        num_different_pieces = 3
        return num_resources + num_different_pieces
    
    def reset(self, seed=None, return_info=False, options=False):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.game_board = GameBoard()
        self.game_board.set_screen_dimensions(1400, 700)
        self.game_rules = GameRules(self.game_board)
        self.players = [
            Player(color=(255, 0, 0), settlements=5, roads=15, cities=4), # Red player
            Player(color=(0, 0, 255), settlements=5, roads=15, cities=4), # Blue player
            Player(color=(20, 220, 20), settlements=5, roads=15, cities=4), # Green player
            Player(color=(255, 165, 0), settlements=5, roads=15, cities=4) # Orange player
        ]
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_board.generate_board(board_radius=2)
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self._agent_selector.next()
        
        self.last_victory_points = {agent: 0 for agent in self.agents}
        
        if return_info:
            return self.observe(self.agent_selection), {}
        else:
            return self.observe(self.agent_selection)
        
    def observe(self, agent):
        board_state = self.get_board_state()
        player_state = self.get_player_state(agent)
        enemy_state = self.get_enemy_state(agent)
        
        observation = {
            'board_state': board_state,
            'player_state': player_state,
            'enemy_state': enemy_state,
            'action_mask': self.get_action_mask(agent)
        }
        return observation
    
    def get_action_mask(self, agent):
        valid_actions = self.get_valid_actions(agent)
        action_mask = np.zeros(self.calculate_action_space_size(), dtype=np.float32)
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
        vertex_states = np.array(vertex_states).flatten()
                
        edge_states = []
        for edge in self.game_board.edges.values():
            if edge.road:
                edge_state = np.array([1], dtype=np.float32)
            else:
                edge_state = np.array([0], dtype=np.float32)
            edge_states.append(edge_state)
        edge_states = np.array(edge_states).flatten()
        
        tile_states = []
        resource_to_idx = {'wood': 0, 'brick': 1, 'sheep': 2, 'wheat': 3, 'ore': 4}
        for tile in self.game_board.tiles.values():
            resource_one_hot = np.zeros(6, dtype=np.float32)
            tile_number = tile.number / 12
            resource_one_hot[5] = tile_number
            resource_idx = resource_to_idx.get(tile.resource, -1)
            if resource_idx >= 0:
                resource_one_hot[resource_idx] = 1
            tile_states.append(resource_one_hot)
        tile_states = np.array(tile_states).flatten()
        
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
        
        player_state = np.concatenate([resources, remaining_pieces, victory_points])
        return player_state
    
    def get_enemy_state(self, agent):
        enemy_states = []
        for enemy in self.agents:
            player_idx = self.agent_name_mapping[enemy]
            player = self.players[player_idx]
            if enemy != agent:
                enemy_total_resources = sum(player.resources.values()) / (25 * 5)
                enemy_vp = player.victory_points / 10
                enemy_state = np.array([enemy_total_resources, enemy_vp], dtype=np.float32)
                enemy_states.append(enemy_state)
                
        return np.array(enemy_states, dtype=np.float32)
    
    def step(self, action):
        
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        
        self._cumulative_rewards[agent] = 0.0
        
        action_type, action_param = self.decode_action(action)
        valid_actions = self.get_valid_actions(agent)
        
        if action not in valid_actions:
            self.rewards[agent] = -1.0
            self.terminations[agent] = True
        else:
            if action_type == 'place_settlement':
                vertex_idx = action_param
                vertex = list(self.game_board.vertices.values())[vertex_idx]
                self.game_manager.current_player_index = player_idx
                self.game_manager.place_house(vertex)
            elif action_type == 'place_road':
                edge_idx = action_param
                edge = list(self.game_board.edges.values())[edge_idx]
                self.game_manager.current_player_index = player_idx
                self.game_manager.place_road(edge)
            # Implement city building later, or other actions
            
            reward = self.calculate_reward(agent)
            self.rewards[agent] = reward
            
            self.truncations = {
                agent: self.game_manager.turn >= self.game_manager.max_turns for agent in self.agents
            }
                        
        if self.game_manager.game_over:
            self.terminations = {agent: True for agent in self.agents}
            
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        
        if self.render_mode == "human":
            self.render()
        
    
    def decode_action(self, action):
        num_vertices = len(self.game_board.vertices)
        if 0 <= action < num_vertices:
            return ('place_settlement', action)
        else:
            return ('place_road', action - num_vertices)
    
    def get_valid_actions(self, agent):
        valid_actions = []
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        
        num_vertices = len(self.game_board.vertices)
        num_edges = len(self.game_board.edges)
        
        # Check if the player can build a house or road on any of the vertices or edges
        # Implement city building later
        for idx, vertex in enumerate(self.game_board.vertices.values()):
            if (self.game_rules.is_valid_house_placement(vertex, player, self.game_manager.gamestate) and
                player.can_build_settlement()):
                valid_actions.append(idx)
                
        for idx, edge in enumerate(self.game_board.edges.values()):
            if (self.game_rules.is_valid_road_placement(edge, player, self.game_manager.gamestate) and
                player.can_build_road()):
                valid_actions.append(num_vertices + idx)

        return valid_actions
    
    def calculate_reward(self, agent):
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        current_vp = player.victory_points
        reward = current_vp - self.last_victory_points[agent]
        self.last_victory_points[agent] = current_vp
        return reward
    
    def render(self):
        
        if self.render_mode is None:
            gymnasium.logger.warn(
                "Render function is called but render_mode is None"
            )
            return
        
        else:
            gymnasium.logger.info("render mode not implemented yet, skipping")
        
    
    def close(self):
        pass
        
        