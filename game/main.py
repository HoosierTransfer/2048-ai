from game import Game
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random

class dqn:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)

        self.gamma = 0.99           # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.1      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.update_rate = 1000     # Number of steps until updating the target network

        # Construct DQN models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()
    
    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Conv2D(32, 2, padding="same", activation="relu", input_shape=self.state_size))
        model.add(layers.Conv2D(64, 2, padding="same", activation="relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))

        model.compile(loss="mse", optimizer=keras.optimizers.Adam())

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])  # Returns action using policy

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)

game = Game(4)
state_size = (4, 4, 1)
action_size = 4
agent = dqn(state_size, action_size)

episodes = 500
batch_size = 8
skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
total_time = 0   # Counter for total number of steps taken
all_rewards = 0  # Used to compute avg reward over time
done = False

for e in range(episodes):
    total_reward = 0
    game_score = 0
    game.setup()
    state = game.grid.array()
    
    for time in range(20000):
        total_time += 1
        
        # Every update_rate timesteps we update the target network parameters
        if total_time % agent.update_rate == 0:
            agent.update_target_model()
        
        # Transition Dynamics
        action = agent.act(state)
        next_state, reward, done = game.step(action)
        
        # Return the avg of the last 4 frames
        next_state = next_state
        
        # Store sequence in replay memory
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        game_score += reward
        reward -= 1  # Punish behavior which does not accumulate reward
        total_reward += reward
        
        if done:
            all_rewards += game_score
            
            print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                  .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
            
            break
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)