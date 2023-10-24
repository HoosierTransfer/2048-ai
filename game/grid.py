from tile import Tile
import random
import numpy as np

class Grid:
    def __init__(self, size, previous_state=None):
        self.size = size
        self.cells = previous_state if previous_state is not None else self.empty()

    def empty(self):
        cells = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                row.append(None)
            cells.append(row)
        
        return cells

    def from_state(self, state):
        cells = []

        for x in range(self.size):
            row = []
            for y in range(self.size):
                tile = state[x][y]
                row.append(None if tile is None else Tile(tile.position, tile.value))
            cells.append(row)

        return cells
    
    def random_available_cell(self):
        cells = self.available_cells()

        if len(cells) > 0:
            return cells[int(random.random() * len(cells))]
    
    def available_cells(self):
        cells = []

        def callback(x, y, tile):
            if tile is None:
                cells.append((x, y))

        self.each_cell(callback)

        return cells

    def each_cell(self, callback):
        for x in range(self.size):
            for y in range(self.size):
                callback(x, y, self.cells[x][y])
    
    def cells_available(self):
        return len(self.available_cells()) > 0
    
    def cell_available(self, cell):
        return not self.cell_occupied(cell)
    
    def cell_occupied(self, cell):
        return self.cell_content(cell) is not None
    
    def cell_content(self, cell):
        if self.within_bounds(cell):
            return self.cells[cell[0]][cell[1]]
        else:
            return None
    
    def insert_tile(self, tile):
        self.cells[tile.x][tile.y] = tile
    
    def remove_tile(self, tile):
        self.cells[tile.x][tile.y] = None
    
    def within_bounds(self, position):
        return position[0] >= 0 and position[0] < self.size and position[1] >= 0 and position[1] < self.size
    
    def serialize(self):
        cells = []

        for x in range(self.size):
            row = []
            for y in range(self.size):
                tile = self.cells[x][y]
                row.append(None if tile is None else tile.serialize())
            cells.append(row)

        return {
            'size': self.size,
            'cells': cells
        }
    
    def array(self):
        cells = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                row.append(np.log2(self.cells[x][y].value) if self.cells[x][y] is not None else 0)
            
            cells.append(row)

        return np.asarray(cells)

    def __repr__(self):
        cells = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                row.append(self.cells[x][y].value if self.cells[x][y] is not None else 0)
            
            cells.append(row)

        return np.asarray(cells)