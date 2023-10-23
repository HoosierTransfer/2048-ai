from game import Game
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import time

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
