class PrintConsole:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)
        if len(self.logs) > 10000:
            self.logs.pop(0)
        print(message)

    def draw(self, screen):
        pass

    def clear(self):
        self.logs = []

    def get_logs(self):
        return self.logs
