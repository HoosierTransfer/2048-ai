import dearpygui.dearpygui as dpg
import numpy as np
from game import Game
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_array = np.array([
    [2, 4, 8, 16],
    [32, 64, 128, 256],
    [512, 1024, 2048, 2],
    [4, 8, 0, 32]
])

SQUARE_COLORS = {
    0: (205,193,180,255),
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

dpg.create_context()

ai_running = False

frames_per_move = 60

game = Game(4)

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

def make_move():
    global not_moved_counter
    global win
    global game
    global model
    training_state = game.grid.array()

    if not_moved_counter >= 1:
        # increase noise over time
        training_state = gaussian_noise(training_state, 0.01 * not_moved_counter )
        print(0.01 * not_moved_counter)

    state_tensor = tf.convert_to_tensor(training_state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)

    action = tf.argmax(action_probs[0]).numpy()

    state, reward, win, = game.step(action)

    if reward == -4:
        not_moved_counter += 1
    elif not_moved_counter > 0:
        not_moved_counter = 0
    
def start_callback(sender, data):
    global ai_running
    ai_running = True

def stop_callback(sender, data):
    global ai_running
    ai_running = False

def step_callback(sender, data):
    make_move()

def speed_callback(sender, data):
    global frames_per_move
    frames_per_move = data



# Game window
with dpg.window(label="2048 Game") as game_window:
    with dpg.drawlist(width=420, height=420):
        for i in range(4):
            for j in range(4):
                num = game.grid.__repr__()[i, j]
                color = SQUARE_COLORS[num]
                dpg.draw_rectangle((i * 100 + i * 5, j * 100 + j * 5), ((i + 1) * 100 + i * 5, (j + 1) * 100 + j * 5), color=color, fill=color, rounding=7)
                text_size = 20
                text_length = len(str(num))
                if num != 0:
                    dpg.draw_text((i * 100 + i * 5 + 50 - text_length * text_size / 4, j * 100 + j * 5 + 50 - text_size / 2), str(num), color=(0, 0, 0, 255), size=20)


with dpg.window(label="Number Stats", width=300):
    score = dpg.add_text("Score: 0")
    frames_since_move = dpg.add_text("Frames since last move: 0")
    is_guassian_added = dpg.add_text("Is gaussian noise added: False with std: 0")

with dpg.window(label="Controls", width=200):
    dpg.add_button(label="Start", callback=start_callback)
    dpg.add_button(label="Stop", callback=stop_callback)
    dpg.add_button(label="Step", callback=step_callback)
    dpg.add_slider_float(label="Speed", default_value=60, min_value=1, max_value=60, callback=speed_callback)

with dpg.window(label="2048 ai") as main_window:
    pass

# Graph window
# with dpg.window(label="Graph"):
#     pass

# Theme for game window
with dpg.theme() as game_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (187, 173, 160), category=dpg.mvThemeCat_Core)

with dpg.theme() as viewport_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (42, 39, 39), category=dpg.mvThemeCat_Core)

dpg.bind_item_theme(game_window, game_theme)
dpg.bind_item_theme(main_window, viewport_theme)

dpg.create_viewport(title='2048 Game', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window(main_window, True)
frame_count = 0
while dpg.is_dearpygui_running():
    if frame_count % frames_per_move == 0 and ai_running:
        make_move()
    with dpg.window(label="2048 Game"):
        draw()
    dpg.set_value(score, "Score: {}".format(game.score))
    dpg.set_value(frames_since_move, "Frames since last move: {}".format(not_moved_counter))
    dpg.set_value(is_guassian_added, "Is gaussian noise added: {} with std: {}".format(not_moved_counter > 3, 0.01 * not_moved_counter))
    dpg.render_dearpygui_frame()
    frame_count += 1
dpg.destroy_context()
