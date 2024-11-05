
class Player:
    COLOR_NAMES = {
        (255, 0, 0): "Red",
        (0, 0, 255): "Blue",
        (20, 220, 20): "Green",
        (255, 165, 0): "Orange"
    }
    
    
    def __init__(self, color, settlements, roads):
        self.color = color
        self.victory_points = 0
        self.settlements = settlements
        self.roads = roads
        self.resources = {'wood': 4, 'brick': 4, 'sheep': 2, 'wheat': 2, 'ore': 0}
        
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
    
    def get_color(self):
        return self.COLOR_NAMES.get(self.color, str(self.color))
    
    def __str__(self):
        color_name = self.COLOR_NAMES.get(self.color, str(self.color))
        resources_str = ', '.join(f"{resource}: {amount}" for resource, amount in self.resources.items())
        return f"Player {color_name} - VP: {self.victory_points}, {resources_str}"
    
    def __repr__(self):
        return self.__str__()