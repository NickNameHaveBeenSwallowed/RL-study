import numpy as np
import gym

def action_one_hot(action, action_space_size):
    return np.array([[1 if action == i else 0 for i in range(action_space_size)]])

class GymGame:
    def __init__(self, config):
        self.env = gym.make(config.game)
        self.action_space_size = self.env.action_space.n
        self.done = False
        self.observation = self.env.reset()
        self.history = [(self.observation, 0.0)]
        self.render = True

    def step(self, action):
        if self.render:
            self.env.render()
        observation, reward, done, info = self.env.step(action)
        reward = -1 if done else 0
        self.observation = observation
        self.history.append((observation, reward))
        self.done = done

class SelfPlay:
    def __init__(self, config):
        self.config = config

    def build_game(self):
        return GymGame(self.config)

    def play(self, actor):
        actor.initialize()
        game = self.build_game()
        target_list = []
        while not game.done:
            action, policy, value = actor.step(game.observation)
            target_list.append((action_one_hot(action, game.action_space_size), policy, value))
            game.step(action)
        history = game.history
        game.env.close()
        return history, target_list