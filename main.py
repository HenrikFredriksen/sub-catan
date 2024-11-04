import pygame
import numpy as np
from GameBoard import GameBoard
from GameManager import GameManager
from GameRules import GameRules

def main():
    pygame.init()
    screen_width, screen_heigth = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_heigth))
    pygame.display.set_caption("Catan")
    
    board = GameBoard()
    board.set_screen_dimensions(screen_width, screen_heigth)
    rules = GameRules(board)
    manager = GameManager(board, rules)

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
    
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                manager.handle_click(mouse_pos)

        screen.fill((100, 140, 250))
        board.draw(screen)
            
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos_text = f"Mouse Position: {mouse_pos}"
        text_surface = font.render(mouse_pos_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()