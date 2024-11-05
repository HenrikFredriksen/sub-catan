import pygame
import numpy as np
from game.GameBoard import GameBoard
from game.GameManager import GameManager
from game.GameRules import GameRules
from game.Player import Player
from assets.Button import Button

def main():
    pygame.init()
    screen_width, screen_height = 1600, 900
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Catan")
    
    board = GameBoard()
    board.set_screen_dimensions(screen_width, screen_height)
    rules = GameRules(board)
    
     # Initialize players
    player1 = Player(color=(255, 0, 0), settlements=5, roads=15)  # Red player
    player2 = Player(color=(0, 0, 255), settlements=5, roads=15)  # Blue player
    player3 = Player(color=(20, 220, 20), settlements=5, roads=15)  # Green player
    player4 = Player(color=(255, 165, 0), settlements=5, roads=15)  # Orange player
    players = [player1, player2, player3, player4]
    
    manager = GameManager(board, rules, players)

    board_radius = 2
    resources = ["brick", "wood", "sheep", "wheat", "ore"]

    # Generate the tiles in a hex grid
    for q in range(-board_radius, board_radius + 1):
        for r in range(-board_radius, board_radius + 1):
            s = -q - r
            if -board_radius <= s <= board_radius:
                resource = resources[np.random.randint(0, len(resources))]
                number = np.random.choice([2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
                if q == 0 and r == 0:
                    resource = "desert"
                    number = 0
                board.add_tile(resource, number, q, r)

    board.generate_vertices()
    board.generate_edges()
    
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)


    next_turn_button = Button(
        x=10, y=screen_height - 50, width=150, height=40,
        text="Next turn", font=font, color=(200, 200, 200), hover_color=(150, 150, 150),
        action=manager.update
    )
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if not next_turn_button.is_clicked(event):
                    manager.handle_click(mouse_pos)

        screen.fill((100, 140, 250))
        board.draw(screen)
        
        next_turn_button.draw(screen)
        turn_text = f"Turn: {manager.turn}"
        turn_text_surface = font.render(turn_text, True, (255, 255, 255))
        screen.blit(turn_text_surface, (10, screen_height - 80))
            
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos_text = f"Mouse Position: {mouse_pos}"
        mouse_pos_text_surface = font.render(mouse_pos_text, True, (255, 255, 255))
        screen.blit(mouse_pos_text_surface, (10, 10))
        
        y_offset = 30
        for player in players:
            resources_text_surface = font.render(str(player), True, (0,0,0))
            screen.blit(resources_text_surface, (10, y_offset))
            y_offset += 20
        
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()