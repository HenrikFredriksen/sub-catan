import os
import pygame

class Tile:
    def __init__(self, resource, number, position):
        self.resource = resource
        self.number = number
        self.position = position # HexCoordinate
        self.number_size = 60
        
        base_path = os.path.join(os.path.dirname(__file__), '../../img/numbers')
        
        self.number_images = {
            2: pygame.transform.scale(pygame.image.load(os.path.join(base_path, "2b.png")), (self.number_size, self.number_size)),
            3: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"3q.png")), (self.number_size, self.number_size)),
            4: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"4n.png")), (self.number_size, self.number_size)),
            5: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"5o.png")), (self.number_size, self.number_size)),
            6: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"6p.png")), (self.number_size, self.number_size)),
            8: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"8k.png")), (self.number_size, self.number_size)),
            9: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"9m.png")), (self.number_size, self.number_size)),
            10: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"10L.png")), (self.number_size, self.number_size)),
            11: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"11i.png")), (self.number_size, self.number_size)),
            12: pygame.transform.scale(pygame.image.load(os.path.join(base_path,"12h.png")), (self.number_size, self.number_size))
        }
        
    def get_number_image(self):
        return self.number_images.get(self.number)
    
    def get_number(self):
        return self.number
