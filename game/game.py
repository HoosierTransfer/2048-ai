from grid import Grid
from tile import Tile
import random
import numpy as np

class Game:
    def __init__(self, size):
        self.size = size

        self.start_tiles = 2

        self.setup()

    def setup(self):
        self.grid = Grid(self.size)

        self.score = 0
        self.over = False
        self.won = False
        self.keep_playing = False

        self.add_start_tiles()
    
    def add_start_tiles(self):
        for i in range(self.start_tiles):
            self.add_random_tile()
    
    def add_random_tile(self):
        if self.grid.cells_available():
            value = 2 if random.random() < 0.9 else 4
            tile = Tile(self.grid.random_available_cell(), value)

            self.grid.insert_tile(tile)
    
    def serialize(self):
        return {
            "grid": self.grid.serialize(),
            "score": self.score,
            "over": self.over,
            "won": self.won,
            "keepPlaying": self.keep_playing
        }

    def prepare_tiles(self):
        def callback(x, y, tile):
            if tile:
                tile.merged_from = None
                tile.save_position()

        self.grid.each_cell(callback)
    
    def move_tile(self, tile, cell):
        self.grid.cells[tile.x][tile.y] = None
        self.grid.cells[cell[0]][cell[1]] = tile
        tile.update_position(cell)
    
    def move(self, direction):
        vector = self.get_vector(direction)
        traversals = self.build_traversals(vector)
        moved = False

        self.prepare_tiles()

        for x in traversals[0]:
            for y in traversals[1]:
                cell = (x, y)
                tile = self.grid.cell_content(cell)
                if tile is not None:
                    positions = self.find_farthest_position(cell, vector)
                    next = self.grid.cell_content(positions[1])

                    if next and next.value == tile.value and not next.merged_from:
                        merged = Tile(positions[1], tile.value * 2)
                        merged.merged_from = [tile, next]

                        self.grid.insert_tile(merged)
                        self.grid.remove_tile(tile)

                        tile.update_position(positions[1])

                        self.score += merged.value

                        if merged.value == 2048:
                            self.won = True
                    else:
                        self.move_tile(tile, positions[0])

                    if not self.positions_equal(cell, tile):
                        moved = True

        if moved:
            self.add_random_tile()
            if not self.moves_available():
                self.over = True
    
    def get_vector(self, direction):
        vectors = {
            0: (0, -1), # up
            1: (1, 0), # right
            2: (0, 1), # down
            3: (-1, 0) # left
        }

        return vectors[direction]

    def build_traversals(self, vector):
        traversals = [[], []]

        for pos in range(self.size):
            traversals[0].append(pos)
            traversals[1].append(pos)

        if vector[0] == 1:
            traversals[0] = reversed(traversals[0])
        if vector[1] == 1:
            traversals[1] = reversed(traversals[1])
        

        return traversals

    def find_farthest_position(self, cell, vector):
        previous = None

        while True:
            previous = cell
            cell = (previous[0] + vector[0], previous[1] + vector[1])

            if not self.grid.within_bounds(cell) or not self.grid.cell_available(cell):
                break
        
        return (previous, cell)
    
    def moves_available(self):
        return self.grid.cells_available() or self.tile_matches_available()

    def tile_matches_available(self):
        tile = None

        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content((x, y))

                if tile:
                    for direction in range(4):
                        vector = self.get_vector(direction)
                        cell = (x + vector[0], y + vector[1])

                        other = self.grid.cell_content(cell)

                        if other and other.value == tile.value:
                            return True
        
        return False

    def positions_equal(self, first, second):
        return first[0] == second.x and first[1] == second.y

    def step(self, direction):
        score = self.score
        moved = self.move(direction)
        reward = self.score - score
        if not moved:
            reward = -10
        if self.won:
            reward = 1000
        if self.over:
            reward = -1000
        return self.grid.array(), reward, self.won or self.over
    
    
# game = Game(4)
# game.setup()
# print(game.grid)
# print("\n")

# for i in range(10):
#     game.move(random.randint(0, 3))
#     print(game.grid)
#     print("\n")

# print(game.score)