import os
import pygame
import numpy as np
from game.GameBoard import GameBoard
from game.GameManager_pygame import GameManager
from game.GameRules import GameRules
from game.Player import Player
from assets.Button import Button
from assets.Console import Console

class GameLoop:
    def __init__(self):
        self.tile_images = {}
        self.number_images = {}
        self.hex_size = 80
        self.number_size = 45
        

    def main(self):
        pygame.init()
        screen_width, screen_height = 1400, 700
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Catan")

        self.load_resources()
        board = GameBoard(self.tile_images, self.number_images)
        board.set_screen_dimensions(screen_width, screen_height)
        rules = GameRules(board)

         # Initialize players
        self.players = [
            Player(player_id=0, color=(255, 0, 0), settlements=5, roads=15, cities=4, victory_points=0, 
                   resources={'wood': 4,'brick': 4,'sheep': 2,'wheat': 2,'ore': 0}), # Red player
            Player(player_id=1, color=(0, 0, 255), settlements=5, roads=15, cities=4, victory_points=0,
                   resources={'wood': 4,'brick': 4,'sheep': 2,'wheat': 2,'ore': 0}), # Blue player
            Player(player_id=2, color=(20, 220, 20), settlements=5, roads=15, cities=4, victory_points=0,
                   resources={'wood': 4,'brick': 4,'sheep': 2,'wheat': 2,'ore': 0}), # Green player
            Player(player_id=3, color=(255, 165, 0), settlements=5, roads=15, cities=4, victory_points=0,
                   resources={'wood': 4,'brick': 4,'sheep': 2,'wheat': 2,'ore': 0}) # Orange player
            ]

        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
        font_console = pygame.font.SysFont(None, 18)

        console = Console(x=screen_width - 400, y=10, width=390, height=200, font=font_console)
        game_manager = GameManager(board, rules, self.players, console)

        # Generate the tiles in a hex grid
        board_radius = 2
        board.generate_board(board_radius)

        next_turn_button = Button(
            x=10, y=screen_height - 50, width=150, height=40,
            text="Next turn", font=font, color=(200, 200, 200), hover_color=(150, 150, 150),
            action=game_manager.update
        )

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if not next_turn_button.is_clicked(event) or game_manager.game_over:
                        game_manager.handle_click(mouse_pos)

                elif event.type == pygame.MOUSEWHEEL:
                    if event.y < 0:
                        console.scroll("up")
                    elif event.y > 0:
                        console.scroll("down")

            screen.fill((100, 140, 250))
            board.draw_game_loop(screen, game_manager.highlighted_vertecies, game_manager.highlighted_edges)

            next_turn_button.draw(screen)

            turn_text = f"Turn: {game_manager.turn}, Player: {game_manager.current_player.get_color()}"
            turn_text_surface = font.render(turn_text, True, game_manager.current_player.get_color())
            screen.blit(turn_text_surface, (10, screen_height - 80))

            mouse_pos = pygame.mouse.get_pos()
            mouse_pos_text = f"Mouse Position: {mouse_pos}"
            mouse_pos_text_surface = font.render(mouse_pos_text, True, (255, 255, 255))
            screen.blit(mouse_pos_text_surface, (10, 10))

            y_offset = 30
            for player in self.players:
                resources_text_surface = font.render(str(player), True, player.get_color())
                screen.blit(resources_text_surface, (10, y_offset))
                y_offset += 20

            console.draw(screen)

            pygame.display.flip()

        pygame.quit()

    def load_resources(self, hex_size=80, number_size=45):
        base_path = os.path.join(os.path.dirname(__file__), '../../img/')
        self.tile_width = int(hex_size * np.sqrt(3))
        self.tile_height = int(hex_size * 2)
        # Load tile images
        self.tile_images = {
            "wood": pygame.transform.scale(
                pygame.image.load(os.path.join(base_path, 'terrainHexes/', "forest.png")),
                (self.tile_width * 1.02, self.tile_height * 1.02)
            ),
            "brick": pygame.transform.scale(
                pygame.image.load(os.path.join(base_path, 'terrainHexes/', "hills.png")),
                (self.tile_width * 1.02, self.tile_height * 1.02)
            ),
            "sheep": pygame.transform.scale(
                pygame.image.load(os.path.join(base_path, 'terrainHexes/', "pasture.png")),
                (self.tile_width * 1.02, self.tile_height * 1.02)
            ),
            "wheat": pygame.transform.scale(
                pygame.image.load(os.path.join(base_path, 'terrainHexes/', "field.png")),
                (self.tile_width * 1.02, self.tile_height * 1.02)
            ),
            "ore": pygame.transform.scale(
                pygame.image.load(os.path.join(base_path, 'terrainHexes/', "mountain.png")),
                (self.tile_width * 1.02, self.tile_height * 1.02)
            ),
            "desert": pygame.transform.scale(
                pygame.image.load(os.path.join(base_path, 'terrainHexes/', "desert.png")),
                (self.tile_width * 1.02, self.tile_height * 1.02)
            )
        }
        # Load number images
        self.number_images = {}
        for number in range(2, 13):
            if number != 7:
                image_path = os.path.join(base_path, 'numbers/', f"{number}.png")
                self.number_images[number] = pygame.transform.scale(
                    pygame.image.load(image_path),
                    (number_size, number_size)
                )
    

    if __name__ == "__main__":
        main()