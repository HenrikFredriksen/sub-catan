import pygame

class Console:
    def __init__(self, x, y, width, height, font, bg_color=(220,220,200), text_color=(0,0,0)):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color
        self.logs = []
        self.scroll_offset = 0
        

    def log(self, message):
        self.logs.append(message)
        if len(self.logs) > 10000:
            self.logs.pop(0)
        print(message)
        
    def draw(self, screen):
        console_surface = pygame.Surface((self.rect.width, self.rect.height))
        console_surface.fill(self.bg_color)
        
        y_offset = 5
        start_index = max(0, len(self.logs) - 10 - self.scroll_offset)
        end_index = len(self.logs) - self.scroll_offset
        for log in self.logs[start_index:end_index]:
            text_surface = self.font.render(log, True, self.text_color)
            console_surface.blit(text_surface, (5, y_offset))
            y_offset += self.font.get_height() + 5
        
        screen.blit(console_surface, self.rect.topleft)
            
    def scroll(self, direction):
        if direction == "up":
            self.scroll_offset = max(0, self.scroll_offset - 1)
        elif direction == "down":
            self.scroll_offset = min(len(self.logs) - 10, self.scroll_offset + 1)
        
    def clear(self):
        self.logs = []

    def get_logs(self):
        return self.logs