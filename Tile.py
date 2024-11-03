import pygame

class Tile:
    def __init__(self, resource, number, position):
        self.resource = resource
        self.number = number
        self.position = position # HexCoordinate
        self.number_size = 60
        self.number_images = {
            2: pygame.transform.scale(pygame.image.load("img/numbers/2b.png"), (self.number_size, self.number_size)),
            3: pygame.transform.scale(pygame.image.load("img/numbers/3q.png"), (self.number_size, self.number_size)),
            4: pygame.transform.scale(pygame.image.load("img/numbers/4n.png"), (self.number_size, self.number_size)),
            5: pygame.transform.scale(pygame.image.load("img/numbers/5o.png"), (self.number_size, self.number_size)),
            6: pygame.transform.scale(pygame.image.load("img/numbers/6p.png"), (self.number_size, self.number_size)),
            8: pygame.transform.scale(pygame.image.load("img/numbers/8k.png"), (self.number_size, self.number_size)),
            9: pygame.transform.scale(pygame.image.load("img/numbers/9m.png"), (self.number_size, self.number_size)),
            10: pygame.transform.scale(pygame.image.load("img/numbers/10L.png"), (self.number_size, self.number_size)),
            11: pygame.transform.scale(pygame.image.load("img/numbers/11i.png"), (self.number_size, self.number_size)),
            12: pygame.transform.scale(pygame.image.load("img/numbers/12h.png"), (self.number_size, self.number_size))
        }
        
    def get_number_image(self):
        return self.number_images.get(self.number)
