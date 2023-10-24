from game import Game
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import pickle
import os
import gc
import pygame

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

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 128  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

num_actions = 4

game = Game(4)

def create_q_model():
    inputs = layers.Input(shape=(4, 4, 1))
    x = layers.Conv2D(32, 2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)
    action = layers.Dense(num_actions, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=action)

def save_model(model, model_target, episode, optimizer):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/episode_{}'.format(episode)):
        os.makedirs('models/episode_{}'.format(episode))
    model.save_weights('models/episode_{}/weights.h5'.format(episode))
    model.save_weights('models/episode_{}/target_weights.h5'.format(episode))
    symbolic_weights = getattr(optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open('models/episode_{}/optimizer.pkl'.format(episode), 'wb') as f:
        pickle.dump(weight_values, f)
    args = np.asarray([action_history, state_history, state_next_history, rewards_history, done_history, episode_reward_history, running_reward, episode_count, frame_count, epsilon], dtype=object)
    np.save('models/episode_{}/args.npy'.format(episode), args)
    
        

model = create_q_model()

model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()
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
running = True
while True:  # Run until solved
    gc.collect()
    tf.keras.backend.clear_session()
    game.setup()
    state = np.array(game.grid.array())
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # env.render(); Adding this line would show the attempts
        update_display(np.exp2(state))
        # of the agent in a pop up window.
        frame_count += 1
        print("Frame count: {}".format(frame_count))

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            print(game.grid.array())



        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done = game.step(action)
        print("Reward: {}".format(reward))

        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)
    if (episode_count + 1) % 20 == 0:
        save_model(model, model_target, episode_count, optimizer)
        print('Model saved at episode {}'.format(episode_count))

    episode_count += 1

    # if running_reward > 40:  # Condition to consider the task solved
    #     print("Solved at episode {}!".format(episode_count))
    #     break

