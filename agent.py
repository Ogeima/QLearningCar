import tensorflow as tf
import random
from collections import deque

from env import Env, test_env

LEARNING_RATE = 1e-3

class Memory:
    """
        Stocke toutes les étapes de transition d'une partie
    """
    def __init__(self, max_length):
        self.memory = deque([], maxlen = max_length)

    def add(self, *args):
        self.memory.append(*args)

    def random_batch(self, batch_size):
        return random.sample(self.memory, batch_size)


class Agent():
    def __init__(self, n_input, n_output):
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(n_input,)),
            tf.keras.layers.Dense(32, activation = tf.keras.activations.relu, use_bias = True, name = "first_layer"),
            tf.keras.layers.Dense(16, activation = tf.keras.activations.relu, use_bias = True, name = "second_layer"),
            tf.keras.layers.Dense(n_output, activation = tf.keras.activations.linear, use_bias = True, name = "output") # Le modèle ne renvoit pas une proba mais une estimation des récompenses attendues, l'activation est donc linéairek
        ])
        self.model.summary()

        self.memory = Memory(10000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def create_suitable_inputs(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 800

    @tf.function
    def choose_action(self, state):
        return tf.math.argmax(self.model(state), axis = 1)[0]

def test_model():
    agent = Agent(5, 3)
    test_env(agent)


if __name__ == "__main__":
    test_model()
