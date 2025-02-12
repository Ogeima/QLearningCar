import tensorflow as tf
import random
from collections import deque

from env import Env, test_env, Action

LEARNING_RATE = 1e-3
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
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(n_input,)),
            tf.keras.layers.Dense(32, activation = tf.keras.activations.relu, use_bias = True, name = "first_layer"),
            tf.keras.layers.Dense(16, activation = tf.keras.activations.relu, use_bias = True, name = "second_layer"),
            tf.keras.layers.Dense(n_output, activation = tf.keras.activations.linear, use_bias = True, name = "output") # Le modèle ne renvoit pas une proba mais une estimation des récompenses attendues, l'activation est donc linéairek
        ])
        self.model.summary()

        self.memory = Memory(10000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        self.epsilon = 0.99
        self.minimum_epsilon = 0.05
        self.epsilon_decay = 0.9999

        self.gamma = 0.9


    def create_suitable_inputs(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 800

    @tf.function
    def choose_action(self, state):
        return tf.math.argmax(self.model(state), axis = 1)[0]

    def exploration_vs_exploitation(self, state):
        x = random.random()
        # Decay epsilon for next iteration
        self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay)
        if x > self.epsilon:
            return self.choose_action(self.create_suitable_inputs([state]))
        else:
            return tf.keras.random.randint(shape=[], minval = 0, maxval = 3)


    def train(self, env: Env):
        print("training...")
        number_episodes = 600

        for ep in range(number_episodes):
            state = env.initialize()

            while True:
                action = int(self.exploration_vs_exploitation(state))
                a = [Action.LEFT, Action.FORWARD, Action.RIGHT][action]

                reward, next_state, done = env.step(a)
                self.memory.add((state, action, reward, next_state))

                if done == 1: break
                
            # Training when memory is large enough
            if len(self.memory) > BATCH_SIZE:
                game_steps = self.memory.random_batch(BATCH_SIZE)

                with tf.GradientTape() as tape: # Automatic gradients by tensorflow <3
                    q1 = self.model(self.create_suitable_inputs([x[0] for x in game_steps]))

                    rewards = tf.convert_to_tensor([x[2] for x in game_steps])
                    actions = tf.convert_to_tensor([x[1] for x in game_steps])

                    q1_action = tf.cast(tf.gather(q1,actions,axis=1,batch_dims=1), dtype=tf.float64)

                    # Bellman equation ig?? x)
                    q2 = self.model(self.create_suitable_inputs([x[3] for x in game_steps]))
                    real_q1 = tf.cast(rewards, dtype=tf.float64)\
                        + tf.cast(self.gamma*0.1, dtype=tf.float64) * tf.cast(tf.math.argmax(q2, axis=1), dtype=tf.float64)

                    loss = tf.reduce_mean(tf.math.pow(q1_action - real_q1, 2))
                    # print(loss)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def test_model():
    agent = Agent(5, 3)
    env = Env()
    # agent.train(env)
    test_env(agent)


if __name__ == "__main__":
    test_model()
