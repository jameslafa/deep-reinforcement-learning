import gym
import numpy as np

from agent import Agent


class CartPole:
    def __init__(self):
        # Number of steps we select to learn from while replaying from memory
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make('CartPole-v1')
        # Configure model based on the environment
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        # Initialize the Agent
        self.agent = Agent(self.state_size, self.action_size)

    def run(self, render: bool = False):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                index = 0
                while not done:
                    if render:
                        self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run(render=False)
