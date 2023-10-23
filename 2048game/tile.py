class Tile:
    def __init__(self, position, value):
        self.x = position[0]
        self.y = position[1]
        self.position = position
        self.value = 2 if value == 0 or value is None else value

        self.previous_position = None
        self.merged_from = None

    def save_position(self):
        self.previous_position = (self.x, self.y)

    def update_position(self, position):
        self.x = position[0]
        self.y = position[1]

    def serialize(self):
        return {
            'position': (self.x, self.y),
            'value': self.value
        }
