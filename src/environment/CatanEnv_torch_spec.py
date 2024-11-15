from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import gymnasium
from gymnasium import spaces
import numpy as np

from game.GameBoard import GameBoard
from game.GameManager_env import GameManager
from game.GameRules import GameRules
from game.Player import Player
from assets.PrintConsole import PrintConsole
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
    metadata = {'render.modes': ['human'], 'name': 'catan_v0'}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.agents = ['player_1', 'player_2', 'player_3', 'player_4']
        self.possible_agents = self.agents[:]
        self.starting_agents = self.agents + self.agents[::-1]
        self.agent_name_mapping = dict(zip(self.agents, (range(len(self.agents)))))        
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
                
        self.render_mode = render_mode
        
        self.game_manager.gamestate = 'settle_phase'
        
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
        
        return num_vertices * 2 + num_edges * 1 + num_tiles * 6
    
    def calculate_player_state_size(self):
        num_resources = 5
        num_different_pieces = 3
        victory_points = 1
        return num_resources + num_different_pieces + victory_points
    
    def reset(self, seed=None, return_info=False, options=False):
        # reset counter and flags
        self.step_count = 0
        
        # reset agents and selector
        self.agents = ['player_1', 'player_2', 'player_3', 'player_4']
        self.possible_agents = self.agents[:]
        self.starting_agents = self.agents + self.agents[::-1]
        self._agent_selector = CustomAgentSelector(self.starting_agents)
        self.agent_selection = self.agents[0]
        
        # reset game state dictionaries
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.console.log(f"Starting new game with agents: {self.agents}")
        self.console.log(f"Terminations: {self.terminations}")

        self.agent_name_mapping = dict(zip(self.agents, (range(len(self.agents)))))

        # reset game components
        self.game_board = GameBoard()
        self.game_board.set_screen_dimensions(1400, 700)
        self.game_rules = GameRules(self.game_board)
        self.players = [
            Player(color=(255, 0, 0), settlements=5, roads=15, cities=4), # Red player
            Player(color=(0, 0, 255), settlements=5, roads=15, cities=4), # Blue player
            Player(color=(20, 220, 20), settlements=5, roads=15, cities=4), # Green player
            Player(color=(255, 165, 0), settlements=5, roads=15, cities=4) # Orange player
        ]

        # reset game manager and generate new board
        self.game_manager = GameManager(self.game_board, self.game_rules, self.players, self.console)
        self.game_board.generate_board(board_radius=2)
        self.vertices_list = list(self.game_board.vertices.values())
        self.edges_list = list(self.game_board.edges.values())
        #self._reset_game()

        self.game_manager.gamestate = 'settle_phase'
        self.game_manager.dice_rolled = False

        if self.render_mode == "human":
            self.render()
                 
        obs = self.observe(self.agent_selection)
        return obs
    
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
        resource_to_idx = {'wood': 0, 'brick': 1, 'sheep': 2, 'wheat': 3, 'ore': 4}
        for tile in self.game_board.tiles.values():
            resource_one_hot = np.zeros(6, dtype=np.float32)
            tile_number = tile.number / 12
            resource_one_hot[5] = tile_number
            resource_idx = resource_to_idx.get(tile.resource, -1)
            if resource_idx >= 0:
                resource_one_hot[resource_idx] = 1
            tile_states.append(resource_one_hot)
        # tile_states should be number of tiles x 6, 5 for resources and 1 for number
        tile_states = np.array(tile_states).flatten()
        
        # board_state should be number of vertices x 2 + number of edges x 1 + number of tiles x 6
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
      
    #def step(self, action):
    #    print(f"gamestate: {self.game_manager.gamestate}, turn: {self.game_manager.turn}, step: {self.step_count}, action: {action}")   
    #    self.step_count += 1
    #    
    #    agent = self.agent_selection
    #    player_idx = self.agent_name_mapping[agent]
    #    player = self.players[player_idx]
    #    self.game_manager.current_player_index = player_idx
#
    #    if self.game_manager.check_if_game_ended():
    #        self.terminations[agent] = True
    #        self.rewards[agent] = self.calculate_reward(agent)
    #        self.infos[agent]['reason'] = 'Victory'
    #        print(f"step_Agent: {agent} has won the game!")
    #        
    #    elif self.game_manager.game_over:
    #        self.truncations[agent] = True
    #        self.rewards[agent] = self.calculate_reward(agent)
    #        self.infos[agent]['reason'] = 'Game Over'
    #        print(f"step_Game Over!")
    #        
    #    else:
    #        pass
    #    
    #    
    #    if self.phase_transition:
    #        self.phase_transition = False
    #        self.game_manager.gamestate = 'normal_phase'
    #        self.in_starting_phase = False
    #        self._agent_selector = agent_selector(self.agents)
    #        self.agent_selection = self.agents[0]
    #        return
    #    
    #    if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
    #        self._was_dead_step(action)
    #        return
    #    
    #            
    #    #pick the action for the agent from the action dict
    #    if isinstance(action, dict):
    #        action = action.get(agent, None)
    #        if action is None:
    #            raise ValueError(f"No action passed for agent {agent}")
    #        
    #        
    #    valid_actions = self.get_valid_actions(agent)
    #    print(f"Available house/city locations: {len(self.game_manager.highlighted_vertecies)}")
    #    print(f"Available road locations: {len(self.game_manager.highlighted_edges)}")
    #    
    #    if action not in valid_actions:
    #        self.rewards[agent] = -1.0
    #        self.terminations[agent] = True
    #        self.console.log(f"Invalid action: {action}")
    #    else:
    #        action_type, action_param = self.decode_action(action)
#
    #        if action_type == 'pass':
    #            self.game_manager.pass_turn()
    #            self.rewards[agent] = -0.1
    #        elif action_type == 'roll_dice':
    #            self.game_manager.roll_phase()
    #            if self.phase_transition:
    #                self.phase_transition = False
    #            self.rewards[agent] = 0.0
    #            # IDEA: reward for getting resources? penalty for not getting resources?
    #        elif action_type == 'place_settlement':
    #            vertex_idx = action_param
    #            vertex = self.vertices_list[vertex_idx]
    #            self.game_manager.place_house(vertex)
    #            self.rewards[agent] = 1.0
    #        
    #        elif action_type == 'place_city':
    #            vertex_idx = action_param
    #            vertex = self.vertices_list[vertex_idx]
    #            self.game_manager.place_city(vertex)
    #            self.rewards[agent] = 1.0
    #        
    #        elif action_type == 'place_road':
    #            edge_idx = action_param
    #            edge = self.edges_list[edge_idx]
    #            self.game_manager.place_road(edge)
    #            self.rewards[agent] = 0.5
#
    #        reward = self.calculate_reward(agent)
    #        self.rewards[agent] = reward
#
    #        self.game_manager.check_if_game_ended()
    #        if self.game_manager.game_over:
    #            self.game_manager.game_over = False
    #            self.terminations = {agent: True for agent in self.agents}
    #            self.truncations = {agent: False for agent in self.agents}
    #            
    #        self.truncations = {
    #            agent: self.game_manager.turn >= self.game_manager.max_turns for agent in self.agents
    #        }
    #        
    #    self._accumulate_rewards()
    #    
    #    if (self.in_starting_phase and 
    #        self.game_manager.starting_sub_phase == 'house' and
    #        self.game_manager.has_placed_piece):
    #        
    #        self.agent_selection = self._agent_selector.next()
    #        if self.agent_selection is None:
    #            # Starting phase is over
    #            self.game_manager.gamestate = 'normal_phase'
    #            self.phase_transition = True
    #            self.game_manager.dice_rolled = False
    #            self.agent_selection = self.agents[0]
    #            self.terminations = {agent: False for agent in self.agents}
    #            return         
    #               
    #    elif (self.in_starting_phase and 
    #          self.game_manager.starting_sub_phase == 'road'
    #          and self.game_manager.has_placed_piece):
    #        self.agent_selection = agent
    #    else:
    #        if self.game_manager.is_turn_over():
    #            self.game_manager.player_passed_turn = False
    #            self.agent_selection = self._agent_selector.next()
    #        else:
    #            self.agent_selection = agent
    #                
    #    if self.render_mode == "human":
    #        self.render()
        
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
        if self.was_placement_successful:
            if action_type == 'place_house':
                return 5
            elif action_type == 'place_city':
                return 6
            elif action_type == 'place_road':
                return 2
            elif action_type == 'pass_turn':
                return -0.1
        else:
            return -1.0
    
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
        
        