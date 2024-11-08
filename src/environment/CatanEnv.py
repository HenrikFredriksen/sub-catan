from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np

from game.GameBoard import GameBoard
from game.GameManager import GameManager
from game.GameRules import GameRules
from game.Player import Player

class CatanEnv(AECEnv):
    metadata = {'render.modes': ['human'], 'name': 'catan_v0'}
    
    def __init__(self):
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
        self.console = None
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_board.generate_board(board_radius=2)
        
        self.action_spaces = {agent: spaces.Discrete(self.calculate_action_space_size()) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Dict({
            'board_state': spaces.Box(low=0, high=1, shape=(self.calculate_board_state_size(),), dtype=np.float32), 
            'player_state': spaces.Box(low=0, high=1, shape=(self.calculate_player_state_size(),), dtype=np.float32)
        }) for agent in self.agents}
        
    def calculate_action_space_size(self):
        num_vertices = len(self.game_board.vertices)
        num_edges = len(self.game_board.edges)
        return num_vertices + num_edges
    
    def calculate_board_state_size(self):
        num_vertices = len(self.game_board.vertices)
        num_edges = len(self.game_board.edges)
        num_tiles = len(self.game_board.tiles)
        
        return num_vertices * 2 + num_edges * 2 + num_tiles * 2
    
    def calculate_player_state_size(self):
        num_resources = 5
        num_different_pieces = 3
        return num_resources + num_different_pieces
    
    def reset(self, seed=None, return_info=False, option=False):
        self.agents = self.possible_agents[:]
        self._agent_selector.reset()
        self.game_board = GameBoard()
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
        self.dones = {agent: False for agent in self.agents}
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
        
        obsertvation = {
            'board_state': board_state,
            'player_state': player_state
        }
        return obsertvation
    
    def get_board_state(self):
        # - For each vertex: [has_settlement, has_city]
        # - For each edge: [has_road]
        # - For each tile: [resource_type_one_hot]
        
        board_state = np.zeros(self.calculate_board_state_size(), dtype=np.float32)
        
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
        
        player_state = np.concatenate([resources, remaining_pieces])
        return player_state
    
    def step(self, action):
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        
        action_type, action_param = self.decode_action(action)
        valid_actions = self.get_valid_actions(agent)
        
        
        if action not in valid_actions:
            # Invalid action
            pass
        else:
            if action_type == 'place_settlement':
                vertex_idx = action_param
                vertex = self.game_board.vertices.values()[vertex_idx]
                self.game_manager.current_player_index = player_idx
                self.game_manager.place_house(vertex)
            elif action_type == 'place_road':
                edge_idx = action_param
                edge = self.game_board.edges.values()[edge_idx]
                self.game_manager.current_player_index = player_idx
                self.game_manager.place_road(edge)
            # Implement city building later, or other actions
            
            reward = self.calculate_reward(agent)
            self.rewards[agent] = reward
            
        if self.game_manager.game_over:
            self.dones = {agent: True for agent in self.agents}
            
        self.agent_selection = self._agent_selector.next()
    
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
            if self.game_rules.is_valid_house_placement(vertex, player, self.game_manager.gamestate):
                valid_actions.append(idx)
                
        for idx, edge in enumerate(self.game_board.edges.values()):
            if self.game_rules.is_valid_road_placement(edge, player, self.game_manager.gamestate):
                valid_actions.append(num_vertices + idx)

        return valid_actions
    
    def calculate_reward(self, agent):
        player_idx = self.agent_name_mapping[agent]
        player = self.players[player_idx]
        current_vp = player.victory_points
        reward = current_vp - self.last_victory_points[agent]
        self.last_victory_points[agent] = current_vp
        return reward
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
        
        