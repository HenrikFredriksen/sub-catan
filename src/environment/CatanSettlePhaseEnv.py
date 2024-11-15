from environment.CatanEnv_torch_spec import CatanEnv

class CatanSettlePhaseEnv(CatanEnv):
    def __init__(self):
        super().__init__()
        self.max_settlements = len(self.possible_agents) * 2

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

    def reset(self):
        obs = super().reset()
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        return obs