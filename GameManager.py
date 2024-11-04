from House import House

class GameManager:
    def __init__(self, game_board, game_rules):
        self.game_board = game_board
        self.game_rules = game_rules
        self.current_player = 1
        self.game_over = False
        
    def handle_click(self, mouse_pos):
        print(f"Mouse clicked at: {mouse_pos}")
        nearest_vertex = self.game_board.find_nearest_vertex(mouse_pos)
        if nearest_vertex:
            print(f"Nearest vertex: {nearest_vertex.position}")
            if self.game_rules.is_valid_house_placement(nearest_vertex):
                house = House(vertex=nearest_vertex, player=self.current_player)
                nearest_vertex.house = house
                print(f"Placed house at {nearest_vertex.position}")
            else:
                print("Invalid placement")
                