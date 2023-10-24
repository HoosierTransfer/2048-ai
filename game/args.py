import numpy as np

action_history, state_history, state_next_history, rewards_history, done_history, episode_reward_history, running_reward, episode_count, frame_count, epsilon  = np.load('args.npy', allow_pickle=True)

print(epsilon)