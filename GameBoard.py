import HexCoordinate as pos
from Tile import Tile
import pygame
import numpy as np

class GameBoard:
    def __init__(self):
        self.hex_size = 100
        self.tiles = {}
        # Tile dimensions for hex grid, width to height ratio is sqrt(3)/2
        self.tile_width = int(self.hex_size * np.sqrt(3))
        self.tile_height = int(self.hex_size * 2)
        self.tile_images = {
            "wood": pygame.transform.scale(pygame.image.load("img/terrainHexes/forest.png"), (self.tile_width, self.tile_height)),
            "brick": pygame.transform.scale(pygame.image.load("img/terrainHexes/hills.png"), (self.tile_width, self.tile_height)),
            "sheep": pygame.transform.scale(pygame.image.load("img/terrainHexes/pasture.png"), (self.tile_width, self.tile_height)),
            "wheat": pygame.transform.scale(pygame.image.load("img/terrainHexes/field.png"), (self.tile_width, self.tile_height)),
            "ore": pygame.transform.scale(pygame.image.load("img/terrainHexes/mountain.png"), (self.tile_width, self.tile_height)),
            "desert": pygame.transform.scale(pygame.image.load("img/terrainHexes/desert.png"), (self.tile_width, self.tile_height))
        }
        
    def add_tile(self, resource, number, q, r):
        position = pos.HexCoordinate(q, r)
        tile = Tile(resource, number, position)
        self.tiles[position] = tile
        
    def get_tile(self, q, r):
        position = pos.HexCoordinate(q, r)
        return self.tiles[position]
    
    def get_neighboring_tiles(self, q , r):
        position = pos.HexCoordinate(q, r)
        neighbors = position.get_neighbors()
        return [self.tiles.get(neighbor) for neighbor in neighbors if neighbor in self.tiles]
    
    def draw(self, screen):
        for tile in self.tiles.values():
            image = self.tile_images[tile.resource]
            x, y = self.hex_to_pixel(tile.position)
            
            x -= self.tile_width // 2
            y -= self.tile_height // 2
            screen.blit(image, (x, y))
            
    def hex_to_pixel(self, hex_coord):
        size = self.hex_size
        x = size * np.sqrt(3) * (hex_coord.q + hex_coord.r/2)
        y = size * (3/2) * hex_coord.r
        return x + self.screen_width // 2, y + self.screen_height // 2
    
    def set_screen_dimensions(self, width, height):
        self.screen_width = width
        self.screen_height = height
   
pygame.init()
screen_width, screen_heigth = 1200, 1000
screen = pygame.display.set_mode((screen_width, screen_heigth))
pygame.display.set_caption("Catan")
 
board = GameBoard()
board.set_screen_dimensions(screen_width, screen_heigth)

board_radius = 2

resources = ["brick", "wood", "sheep", "wheat", "ore"]

# Generate the tiles in a hex grid
for q in range(-board_radius, board_radius + 1):
    for r in range(-board_radius, board_radius + 1):
        s = -q - r
        if -board_radius <= s <= board_radius:
            resource = resources[(q + r + board_radius * 3) % len(resources)]
            if q == 0 and r == 0:
                resource = "desert"
            number = 0  # Assign numbers as needed
            board.add_tile(resource, number, q, r)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        screen.fill((100, 140, 250))
        board.draw(screen)
        pygame.display.flip()
        
pygame.quit()

