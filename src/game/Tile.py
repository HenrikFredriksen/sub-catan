import os
import pygame

'''
The Tile class represents a tile on the game board.
The tile has a resource, a number, and a position.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Tile:
    def __init__(self, resource, number, position):
        self.resource = resource
        self.number = number
        self.position = position # HexCoordinate
        self.number_size = 45

    def get_number_image(self):
        return self.number_images.get(self.number)
    
    def get_number(self):
        return self.number
