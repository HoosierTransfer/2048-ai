from game import Game
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import pickle
import os
import pygame
import time
# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 4  # 4x4 grid
SQUARE_SIZE = 100
GAP_SIZE = 10
WIDTH = GRID_SIZE * (SQUARE_SIZE + GAP_SIZE) - GAP_SIZE
HEIGHT = GRID_SIZE * (SQUARE_SIZE + GAP_SIZE) - GAP_SIZE
print(WIDTH)
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
            square_color = SQUARE_COLORS.get(value, (205,193,180))
            pygame.draw.rect(screen, square_color, (x, y, SQUARE_SIZE, SQUARE_SIZE), border_radius=7)
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
    screen.fill(BACKGROUND_COLOR)
    draw_grid(new_grid)
    pygame.display.flip()

num_actions = 4
def create_q_model():
    inputs = layers.Input(shape=(4, 4, 1))
    x = layers.Conv2D(32, 2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    x = layers.Dense(256, activation="relu")(x)
    action = layers.Dense(num_actions, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=action)

model = create_q_model()

model.load_weights('weights.h5')

game = Game(4)

not_moved_counter = 0

win = False

def gaussian_noise(x,std):
    noise = np.random.normal(0, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 

state = game.grid.array()
clock=pygame.time.Clock()
time.sleep(1)
pygame.mixer.init()
pygame.mixer.music.load("wait.mp3")
state_tensor = tf.convert_to_tensor(game.grid.array())
state_tensor = tf.expand_dims(state_tensor, 0)
action_probs = model(state_tensor, training=False)
pygame.mixer.music.play(100)
while True:
    game.setup()
    state = game.grid.array()
    while not win:
        pygame.event.get()
        # clock.tick(2)
        training_state = game.grid.array()

        if not_moved_counter >= 1:
            # increase noise over time
            training_state = gaussian_noise(training_state, 0.02 * not_moved_counter)

        state_tensor = tf.convert_to_tensor(training_state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)

        action = tf.argmax(action_probs[0]).numpy()

        state, reward, win, = game.step(action)

        if reward == -4:
            not_moved_counter += 1
        elif not_moved_counter > 0:
            not_moved_counter = 0

        update_display(game.grid.__repr__())
        time.sleep(60/114)

print("The ai has " + "won" if game.won else "lost")
print(game.grid)


while True:
    update_display(game.grid.__repr__())
