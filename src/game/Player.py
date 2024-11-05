
class Player:
    def __init__(self, color, settlements, roads):
        self.color = color
        self.victory_points = 0
        self.settlements = settlements
        self.roads = roads
        self.resources = {'wood': 20, 'brick': 20, 'sheep': 5, 'wheat': 5, 'ore': 0}
        
    def add_resource(self, resource, amount):
        self.resources[resource] += amount
    
    def remove_resource(self, resource, amount):
        self.resources[resource] -= amount
    
    def can_build_settlement(self):
        return (self.settlements > 0 and 
            self.resources['wood'] > 0 and 
            self.resources['brick'] > 0 and 
            self.resources['sheep'] > 0 and 
            self.resources['wheat'] > 0)
    
    def can_build_road(self):
        return (self.roads > 0 and
                self.resources['wood'] > 0 and
                self.resources['brick'] > 0)
    
    def add_victory_points(self, amount):
        self.victory_points += amount
        
    def get_victory_points(self):
        return self.victory_points