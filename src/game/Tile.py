import os
import pygame

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
