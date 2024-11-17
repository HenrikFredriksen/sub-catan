from environment.CatanEnv_torch_spec import CatanEnv
from environment.CustomAgentSelector import CustomAgentSelector
from game.GameBoard import GameBoard
from game.GameManager_env import GameManager
from game.GameRules import GameRules
from game.Player import Player

import numpy as np
import random

class CatanSettlePhaseEnv(CatanEnv):
    def __init__(self, writer=None):
        super().__init__()
        self.max_settlements = len(self.possible_agents) * 2
        self.full_resource_settles = {agent: 0 for agent in self.possible_agents}
        self.writer = writer

    def calculate_reward(self, agent, action_type):
        reward = 0
        if action_type == 'place_house':
            vertex = self.game_manager.last_placed_house_vertex.get(self.agent_selection)

            if vertex:
                adj_tiles = self.game_board.get_tiles_adj_to_vertex(vertex)
                reward += len(adj_tiles) * 2

                resources = self.game_board.get_resources_adj_to_vertex(vertex)
                unique_resources = set(resources.keys()) - {'desert'}
                reward += len(unique_resources) * 3
                
                numbers = self.game_board.get_numbers_adj_to_vertex(vertex)
                for number in numbers.values():
                    if number == 6 or number == 8:
                        reward += 5 * 1.5
                    elif number == 5 or number == 9:
                        reward += 4 * 1.25
                    elif number == 4 or number == 10:
                        reward += 3
                    elif number == 3 or number == 11:
                        reward += 2
                    elif number == 2 or number == 12:
                        reward += 1
                    else:
                        raise ValueError(f"Invalid number {number} adjacent to vertex {vertex.position}")

                player = self.players[self.agent_name_mapping[agent]]
                settlement_count = 5 - player.settlements

                if settlement_count == 2:

                    for v in self.game_board.vertices.values():
                        if (v.house and 
                            v.house.player == player and 
                            v != vertex):
                            first_settle_adj_res = self.game_board.get_resources_adj_to_vertex(v)   
                            total_resources = set(first_settle_adj_res.keys())
                            total_resources.update(unique_resources)
                            total_resources.discard('desert')

                            if len(total_resources) == 5:
                                self.number_of_full_res_settles += 1
                                if self.writer:
                                    self.writer.add_scalar(
                                        f'Full Resource Settles/{agent}',
                                        self.number_of_full_res_settles[agent],
                                        self._episode_steps
                                    )
                                    
                                    total_full_res = sum(self.number_of_full_res_settles.values())
                                    self.writer.add_scalar(
                                        'Full Resource Settles/Total',
                                        total_full_res,
                                        self._episode_steps
                                    )
                                self.console.log(f"Player {self.players[self.agent_name_mapping[agent]]} has placed settlements on all 5 resources")
                                reward += 20
                            break

        elif action_type == 'place_road':
            return 1
        else:
            return -1
        
        return reward
        
    def step(self, action):
        super().step(action)
    
        settlements_placed = sum(5 - player.settlements for player in self.players)
    
        if settlements_placed >= self.max_settlements or self.game_manager.gamestate == 'normal_phase':
            self.terminations = {agent: True for agent in self.possible_agents}
            self.truncations = {agent: True for agent in self.possible_agents}
        
        self.game_manager.has_placed_piece = False
        self.game_manager.dice_rolled = False

    def reset(self, seed=None, return_info=False, options=None):
        obs = super().reset()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.step_count = 0
        
        self.agents = ['player_1', 'player_2', 'player_3', 'player_4']
        self.possible_agents = self.agents[:]
        self.starting_agents = self.agents + self.agents[::-1]
        self._agent_selector = CustomAgentSelector(self.starting_agents)
        self.agent_selection = self.agents[0]
        
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.number_of_full_res_settles = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {'seed': seed} for agent in self.possible_agents}
        
        self.agent_name_mapping = dict(zip(self.agents, range(len(self.agents))))
        
        self.game_board = GameBoard()
        self.game_board.set_screen_dimensions(1400, 700)
        self.game_board.generate_board(board_radius=2)
        
        self.game_rules = GameRules(self.game_board)
        self.players = [
            Player(player_id=0, color=(255, 0, 0), settlements=5, roads=15, cities=4, resources={'wood': 4, 'brick': 4, 'sheep': 2, 'wheat': 2, 'ore': 2}),
            Player(player_id=1, color=(0, 0, 255), settlements=5, roads=15, cities=4, resources={'wood': 4, 'brick': 4, 'sheep': 2, 'wheat': 2, 'ore': 2}),
            Player(player_id=2, color=(20, 220, 20), settlements=5, roads=15, cities=4, resources={'wood': 4, 'brick': 4, 'sheep': 2, 'wheat': 2, 'ore': 2}),
            Player(player_id=3, color=(255, 165, 0), settlements=5, roads=15, cities=4, resources={'wood': 4, 'brick': 4, 'sheep': 2, 'wheat': 2, 'ore': 2})
        ]
        
        self.vertices_list = list(self.game_board.vertices.values())
        self.edges_list = list(self.game_board.edges.values())
        
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_manager.gamestate = 'settle_phase'
        self.game_manager.has_placed_piece = False
        self.was_placement_successful = False
        
        if self.render_mode == 'human':
            self.render()
            
        self.observe(self.agent_selection)
                
        if return_info:
            return obs, self.infos[self.agent_selection]
        return obs