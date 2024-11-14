
class CustomAgentSelector:
    def __init__(self, agent_order):
        self.agent_order = agent_order
        self.current_agent_index = 0
        
    def reset(self):
        self.current_agent_index = 0
        
    def next(self):
        if self.current_agent_index < len(self.agent_order):
            agent_selection = self.agent_order[self.current_agent_index]
            self.current_agent_index += 1
            return agent_selection
        else:
            return None
        
    def is_last(self):
        return self.current_agent_index == len(self.agent_order)