import os
from game.HexCoordinate import HexCoordinate
from game.Tile import Tile
from game.Vertex import Vertex
from game.Edge import Edge
import pygame
import numpy as np

class GameBoard:
    def __init__(self):
        self.hex_size = 80
        self.tiles = {}
        self.vertices = {}
        self.edges = {}
        # Tile dimensions for hex grid, width to height ratio is sqrt(3)/2
        self.tile_width = int(self.hex_size * np.sqrt(3))
        self.tile_height = int(self.hex_size * 2)
        
        base_path = os.path.join(os.path.dirname(__file__), '../../img/terrainHexes')
        self.tile_images = {
            "wood": pygame.transform.scale(pygame.image.load(os.path.join(base_path,"forest.png")), 
                                           (self.tile_width*1.02, self.tile_height*1.02)),
            "brick": pygame.transform.scale(pygame.image.load(os.path.join(base_path,"hills.png")), 
                                            (self.tile_width*1.02, self.tile_height*1.02)),
            "sheep": pygame.transform.scale(pygame.image.load(os.path.join(base_path,"pasture.png")), 
                                            (self.tile_width*1.02, self.tile_height*1.02)),
            "wheat": pygame.transform.scale(pygame.image.load(os.path.join(base_path,"field.png")), 
                                            (self.tile_width*1.02, self.tile_height*1.02)),
            "ore": pygame.transform.scale(pygame.image.load(os.path.join(base_path,"mountain.png")), 
                                          (self.tile_width*1.02, self.tile_height*1.02)),
            "desert": pygame.transform.scale(pygame.image.load(os.path.join(base_path,"desert.png")), 
                                             (self.tile_width*1.02, self.tile_height*1.02))
        }
        
    def add_tile(self, resource, number, q, r):
        position = HexCoordinate(q, r)
        tile = Tile(resource, number, position)
        self.tiles[position] = tile
        
    def get_tile(self, q, r):
        position = HexCoordinate(q, r)
        return self.tiles[position]
    
    def get_neighboring_tiles(self, q , r):
        position = HexCoordinate(q, r)
        neighbors = position.get_neighbors()
        print(neighbors)
        return [self.tiles.get(neighbor) for neighbor in neighbors if neighbor in self.tiles]
            
    def hex_to_pixel(self, hex_coord):
        size = self.hex_size
        x = size * np.sqrt(3) * (hex_coord.q + hex_coord.r/2)
        y = size * (3/2) * hex_coord.r
        return x + self.screen_width // 2, y + self.screen_height // 2
    
    def pixel_to_hex(self, x, y):
        size = self.hex_size
        q = (np.sqrt(3)/3 * x  -  1/3 * y) / size
        r = (2/3 * y) / size
        return HexCoordinate(q, r)
    
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
    
    def get_tile_vertices(self, tile):
        corners = self.get_hex_corners(tile.position)
        vertices = []
        for corner in corners:
            corner_int = (int(round(corner[0])), int(round(corner[1])))
            if corner_int in self.vertices:
                vertices.append(self.vertices[corner_int])
        return vertices
    
    # Get the tiles adjacent to a vertex, Inefficient?
    def get_tiles_adj_to_vertex(self, vertex):
        adjacent_tiles = []
        for tile in self.tiles.values():
            if vertex in self.get_tile_vertices(tile):
                adjacent_tiles.append(tile)
        return adjacent_tiles
        
    # Only for debugging
    def draw_grid(self, screen):
        for tile in self.tiles.values():
            corners = self.get_hex_corners(tile.position)
            pygame.draw.polygon(screen, (0, 0, 0), corners, 1)
        
    def draw_vertices(self, screen, highlighted_vertices):
        for vertex in self.vertices.values():
            if vertex in highlighted_vertices:
                pygame.draw.circle(screen, (255, 255, 255), vertex.position, 10)
                
            elif vertex.house:
                pygame.draw.circle(screen, (0,0,0), vertex.position, 10)
                pygame.draw.circle(screen, vertex.house.player.color, vertex.position, 8)
            
            elif vertex.city:
                pygame.draw.rect(screen, (0,0,0), (vertex.position[0] - 10, vertex.position[1] - 10, 20, 20))
                pygame.draw.rect(screen, vertex.city.player.color, 
                                 (vertex.position[0] - 8, vertex.position[1] - 8, 16, 16))
                
    def draw_edges(self, screen, highlighted_edges):
        for edge in self.edges.values():
            if edge in highlighted_edges:
                pygame.draw.line(screen, (255, 255, 255), edge.vertex1.position, edge.vertex2.position, 6)
                
            if edge.road:
                pygame.draw.line(screen, (0, 0, 0), edge.vertex1.position, edge.vertex2.position, 6)
                pygame.draw.line(screen, edge.road.player.color, edge.vertex1.position, edge.vertex2.position, 4)
            
        
    # Draw the tiles, vertices and edges
    def draw(self, screen, highlighted_vertices, highlighted_edges):
        for tile in self.tiles.values():
            image = self.tile_images[tile.resource]
            x, y = self.hex_to_pixel(tile.position)
            
            x -= self.tile_width // 2
            y -= self.tile_height // 2
            screen.blit(image, (x, y))
            
            number_image = tile.get_number_image()
            if number_image:
                number_rect = number_image.get_rect(center=(x + self.tile_width // 2, 
                                                            y + self.tile_height // 2))
                screen.blit(number_image, number_rect.topleft)
        self.draw_edges(screen, highlighted_edges)
        self.draw_vertices(screen, highlighted_vertices)
        
    # Generate the vertices and edges for the hex grid
    def generate_vertices(self):
        for tile in self.tiles.values():
            corners = self.get_hex_corners(tile.position)
            for i, corner in enumerate(corners):
                corner_int = (int(round(corner[0])), int(round(corner[1])))
                if corner_int not in self.vertices:
                    self.vertices[corner_int] = Vertex(corner_int)
                # Add neighbors
                next_corner = corners[(i + 1) % len(corners)]
                next_corner_int = (int(round(next_corner[0])), int(round(next_corner[1])))
                if next_corner_int not in self.vertices:
                    self.vertices[next_corner_int] = Vertex(next_corner_int)
                self.vertices[corner_int].add_neighbor(self.vertices[next_corner_int])
                self.vertices[next_corner_int].add_neighbor(self.vertices[corner_int])

                    
    def generate_edges(self):
        for vertex in self.vertices.values():
            for neighbor in vertex.neighbors:
                edge_key = tuple(sorted([vertex.position, neighbor.position]))
                if edge_key not in self.edges:
                    self.edges[edge_key] = Edge(vertex, neighbor)

    def set_screen_dimensions(self, width, height):
        self.screen_width = width
        self.screen_height = height
        
    def generate_board(self, board_radius):
        numbers = [2, 3,3, 4,4, 5,5, 6,6, 8,8, 9,9, 10,10, 11,11, 12]
        resources = ["brick"] * 3 + ["ore"] * 3 + ["wood"] * 4 + ["sheep"] * 4 + ["wheat"] * 4 + ["desert"]
        np.random.shuffle(numbers)
        np.random.shuffle(resources)
        numbers_index = 0
        resources_index = 0
        for q in range(-board_radius, board_radius + 1):
            for r in range(-board_radius, board_radius + 1):
                s = -q - r
                if -board_radius <= s <= board_radius:
                    resource = resources[resources_index]
                    if resource != "desert":
                        number = numbers[numbers_index]
                        numbers_index += 1
                    else:
                        number = 0
                    resources_index += 1
                    self.add_tile(resource, number, q, r)
        
        self.generate_vertices()
        self.generate_edges()
        
    def find_nearest_vertex(self, mouse_pos, proximity_radius):
        nearest_vertex = None
        min_distance = float("inf")
        for vertex in self.vertices.values():
            vx, vy = vertex.position
            distance = np.hypot(mouse_pos[0] - vx, mouse_pos[1] - vy)
            if distance < min_distance and distance < proximity_radius:
                nearest_vertex = vertex
                min_distance = distance
        return nearest_vertex
    
    def find_nearest_edge(self, mouse_pos, proximity_radius):
        nearest_edge = None
        min_distance = float("inf")
        for edge in self.edges.values():
            mid_x = (edge.vertex1.position[0] + edge.vertex2.position[0]) // 2
            mid_y = (edge.vertex1.position[1] + edge.vertex2.position[1]) // 2
            distance = np.hypot(mouse_pos[0] - mid_x, mouse_pos[1] - mid_y)
            if distance < min_distance and distance < proximity_radius:
                nearest_edge = edge
                min_distance = distance
        return nearest_edge
