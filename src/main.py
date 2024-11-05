import pygame
import numpy as np
from game.GameBoard import GameBoard
from game.GameManager import GameManager
from game.GameRules import GameRules
from game.Player import Player
from assets.Button import Button

def main():
    pygame.init()
    screen_width, screen_height = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Catan")
    
    board = GameBoard()
    board.set_screen_dimensions(screen_width, screen_height)
    rules = GameRules(board)
    
     # Initialize players
    player1 = Player(color=(255, 0, 0), settlements=5, roads=15)  # Red player
    player2 = Player(color=(0, 0, 255), settlements=5, roads=15)  # Blue player
    players = [player1, player2]
    
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


    change_player_button = Button(
        x=10, y=screen_height - 50, width=150, height=40,
        text="Change Player", font=font, color=(200, 200, 200), hover_color=(150, 150, 150),
        action=manager.change_player
    )
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if not change_player_button.is_clicked(event):
                    manager.handle_click(mouse_pos)

        screen.fill((100, 140, 250))
        board.draw(screen)
        
        change_player_button.draw(screen)
            
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos_text = f"Mouse Position: {mouse_pos}"
        mouse_pos_text_surface = font.render(mouse_pos_text, True, (255, 255, 255))
        screen.blit(mouse_pos_text_surface, (10, 10))
        
        resources_text_p1 = f"Player 1: {player1.resources}"
        resources_text_p2 = f"Player 2: {player2.resources}"
        resources_text_surface_p1 = font.render(resources_text_p1, True, (0,0,0))
        resources_text_surface_p2 = font.render(resources_text_p2, True, (0,0,0))
        screen.blit(resources_text_surface_p1, (screen_width // 4, 10))
        screen.blit(resources_text_surface_p2, (screen_width // 4, screen_height - 30))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()