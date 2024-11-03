import HexCoordinate as pos
from Tile import Tile
from Vertex import Vertex
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
        self.vertices = {}
        
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
        print(neighbors)
        return [self.tiles.get(neighbor) for neighbor in neighbors if neighbor in self.tiles]
            
    def hex_to_pixel(self, hex_coord):
        size = self.hex_size
        x = size * np.sqrt(3) * (hex_coord.q + hex_coord.r/2)
        y = size * (3/2) * hex_coord.r
        return x + self.screen_width // 2, y + self.screen_height // 2
    
    def get_hex_corners(self, hex_coord):
        corners = []
        center_x, center_y = self.hex_to_pixel(hex_coord)
        for i in range(6):
            # 30 degrees offset to start at the top for pointy top hexes
            angle_deg = 60 * i + 30 
            angle_rad = np.pi / 180 * angle_deg
            x = center_x + self.hex_size * np.cos(angle_rad)
            y = center_y + self.hex_size * np.sin(angle_rad)
            corners.append((x, y))
        return corners
    
    def draw_grid(self, screen):
        for tile in self.tiles.values():
            corners = self.get_hex_corners(tile.position)
            pygame.draw.polygon(screen, (0, 0, 0), corners, 1)
            
    def draw_vertices(self, screen):
        for vertex in self.vertices.values():
            if vertex.house:
                pygame.draw.circle(screen, (0, 255, 0), vertex.position, 8)
            else:
                pygame.draw.circle(screen, (255, 0, 0), vertex.position, 5)
        
    def draw(self, screen):
        for tile in self.tiles.values():
            image = self.tile_images[tile.resource]
            x, y = self.hex_to_pixel(tile.position)
            
            x -= self.tile_width // 2
            y -= self.tile_height // 2
            screen.blit(image, (x, y))
            
            number_image = tile.get_number_image()
            if number_image:
                number_rect = number_image.get_rect(center=(x + self.tile_width // 2, y + self.tile_height // 2))
                screen.blit(number_image, number_rect.topleft)
        self.draw_grid(screen)
        self.draw_vertices(screen)
        
    def generate_vertices(self):
        for tile in self.tiles.values():
            corners = self.get_hex_corners(tile.position)
            for corner in corners:
                corner_int = (int(round(corner[0])), int(round(corner[1])))
                if corner_int not in self.vertices:
                    self.vertices[corner_int] = Vertex(corner_int)

    def set_screen_dimensions(self, width, height):
        self.screen_width = width
        self.screen_height = height
        
    def find_nearest_vertex(self, mouse_pos):
        nearest_vertex = None
        min_distance = float("inf")
        for vertex in self.vertices.values():
            vx, vy = vertex.position
            distance = np.hypot(mouse_pos[0] - vx, mouse_pos[1] - vy)
            if distance < min_distance and distance < self.hex_size / 2:
                nearest_vertex = vertex
                min_distance = distance
        return nearest_vertex

