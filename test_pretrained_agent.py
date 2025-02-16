import tensorflow as tf

from env import test_env
from agent import Agent

def play():
    agent = Agent(5, 3)
    agent.model = tf.keras.models.load_model("weights/trained_model2.h5")
    test_env(agent)

if __name__ == "__main__":
    play()
