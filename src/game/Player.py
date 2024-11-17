'''
Player class

This class represents a player in the game. 
It keeps track of the player's color, resources, settlements, roads, cities, and victory points. 
It also provides methods to change the state of the player, such as:
building:
- settlements
- roads
- cities

resource management:
- trade with the bank
- add or remove resources.

@author: Henrik Tobias Fredriksen
@date: 17. October 2024
'''
class Player:
    COLOR_NAMES = {
        (255, 0, 0): "Red",
        (0, 0, 255): "Blue",
        (20, 220, 20): "Green",
        (255, 165, 0): "Orange"
    }
    
    
    def __init__(self, 
                 player_id, 
                 color, 
                 settlements=5, 
                 roads=15, 
                 cities=4, 
                 victory_points=0, 
                 resources=None):
        self.player_id = player_id
        self.color = color
        self.victory_points = victory_points
        self.settlements = settlements
        self.roads = roads
        self.cities = cities
        if resources is None:
            self.resources = {'wood': 0,'brick': 0,'sheep': 0,'wheat': 0,'ore': 0}
        else:
            self.resources = resources
        
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
        
    def can_build_city(self):
        return (self.cities > 0 and
                self.resources['wheat'] >= 2 and
                self.resources['ore'] >= 3)
        
    def can_trade_with_bank(self, resource):
        return self.resources[resource] >= 4
    
    def add_victory_points(self, amount):
        self.victory_points += amount
        
    def get_victory_points(self):
        return self.victory_points
    
    def get_color(self):
        return self.COLOR_NAMES.get(self.color, str(self.color))
    
    def __str__(self):
        color_name = self.get_color()
        resources_str = ', '.join(f"{resource}: {amount}" for resource, amount in self.resources.items())
        return f"Player {color_name} - VP: {self.victory_points}, {resources_str}"
    
    def __repr__(self):
        return self.__str__()