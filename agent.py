import tensorflow as tf
import numpy as np
import random
from collections import deque

from env import Env, test_env, Action

LEARNING_RATE = 5e-3
BATCH_SIZE = 128

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

    def __len__(self):
        return len(self.memory)


class Agent():
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.n_input,)),
            tf.keras.layers.Dense(32, activation = tf.keras.activations.elu, name = "first_layer"),
            tf.keras.layers.Dense(16, activation = tf.keras.activations.elu, name = "second_layer"),
            tf.keras.layers.Dense(self.n_output, name = "output"), # Le modèle ne renvoit pas une proba mais une estimation des récompenses attendues, l'activation est donc linéaire
        ])
        self.model.summary()

        self.memory = Memory(10000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss = tf.keras.losses.MeanSquaredError()

        self.gamma = 0.7


    def create_suitable_inputs(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 500

    @tf.function
    def choose_action(self, state):
        return tf.math.argmax(self.model(state), axis = 1)[0]

    def exploration_vs_exploitation(self, state, epsilon):
        x = random.random()
        if x > epsilon:
            return self.choose_action(self.create_suitable_inputs([state]))
        else:
            return np.random.randint(0, self.n_output)

    def train(self, env: Env):
        print("training...")

        number_episodes = 1200

        for ep in range(number_episodes):
            state = env.initialize()
            epsilon = max(1 - ep / number_episodes, 0.01)

            for _ in range(400):
                action = int(self.exploration_vs_exploitation(state, epsilon))
                a = [Action.LEFT, Action.FORWARD, Action.RIGHT][action]

                reward, next_state, done = env.step(a)
                self.memory.add((state, action, reward, next_state, done))

                if done == 1: break
                
            # Training when memory is large enough
            if ep > 50:
                game_steps = self.memory.random_batch(BATCH_SIZE)

                states = [x[0] for x in game_steps]
                next_states = [x[3] for x in game_steps]
                actions = [x[1] for x in game_steps]
                rewards = np.array([x[2] for x in game_steps])
                dones = np.array([x[4] for x in game_steps])

                next_Q_values = self.model(self.create_suitable_inputs(next_states))
                max_next_Q_values = tf.reduce_max(next_Q_values, axis = 1)

                target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)

                action_mask = tf.one_hot(actions, self.n_output)

                with tf.GradientTape() as tape: # Automatic gradients by tensorflow <3
                    all_Q_values = self.model(self.create_suitable_inputs(states))
                    q_values = tf.reduce_sum(all_Q_values * action_mask, axis = 1)
                    loss = self.loss(target_Q_values, q_values)
                    print(loss)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def train_model():
    agent = Agent(5, 3)
    env = Env()
    agent.train(env)
    # agent.model.save("weights/trained_model3.h5")
    print("model_saved")
    del env
    test_env(agent)


if __name__ == "__main__":
    train_model()
