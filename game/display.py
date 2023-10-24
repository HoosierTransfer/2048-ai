import pygame
import sys
import numpy as np
# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 4  # 4x4 grid
SQUARE_SIZE = 100
GAP_SIZE = 10
WIDTH = GRID_SIZE * (SQUARE_SIZE + GAP_SIZE) - GAP_SIZE
HEIGHT = GRID_SIZE * (SQUARE_SIZE + GAP_SIZE) - GAP_SIZE
FONT_SIZE = 36
MAX_NUMBER = 2048

# Colors
BACKGROUND_COLOR = (187, 173, 160)
SQUARE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

# Create a window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grid with Numbers")

# Font for rendering numbers
font = pygame.font.Font(None, FONT_SIZE)

# Function to draw a grid of squares with numbers
def draw_grid(grid):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = col * (SQUARE_SIZE + GAP_SIZE)
            y = row * (SQUARE_SIZE + GAP_SIZE)
            value = grid[row, col]
            square_color = SQUARE_COLORS.get(value, (255, 255, 255))
            pygame.draw.rect(screen, square_color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            if value > 0:
                text_surface = font.render(str(value), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                screen.blit(text_surface, text_rect)

# Function to initialize the display with a 4x4 NumPy array
def init_display(initial_grid):
    screen.fill(BACKGROUND_COLOR)
    draw_grid(initial_grid)
    pygame.display.flip()

# Function to update the display with a new 4x4 NumPy array
def update_display(new_grid):
    draw_grid(new_grid)
    pygame.display.flip()

# # Main game loop
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Example: Initialize the display with a random 4x4 NumPy array
#     if "initialized" not in locals():
#         initial_grid = np.random.choice([0, 2, 4, 8, 16], size=(GRID_SIZE, GRID_SIZE))
#         init_display(initial_grid)
#         initialized = True

#     # Example: Update the display with a new random 4x4 NumPy array on each frame
#     random_grid = np.random.choice([0, 2, 4, 8, 16], size=(GRID_SIZE, GRID_SIZE))
#     update_display(random_grid)

# # Quit Pygame
# pygame.quit()
# sys.exit()