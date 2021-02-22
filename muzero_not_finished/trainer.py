from collections import deque
import tensorflow as tf
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, config):
        self.buffer = deque(maxlen=config.window_size)
        self.config = config

    def save_game(self, history, targets):
        self.buffer.append((history, targets))

    def sample(self):
        sample_size = min(len(self.buffer), self.config.batch_size)
        sample_games = random.sample(self.buffer, sample_size)
        samples = []
        for history, targets in sample_games:
            if len(targets) <= self.config.num_unroll_steps:
                continue

            index = random.randint(0, len(targets) - self.config.num_unroll_steps)

            states = []
            rewards = []
            actions = []
            policys = []
            values = []
            for state, reward in history:
                states.append(state)
                rewards.append(reward)

            for action, policy, value in targets:
                actions.append(action)
                policys.append(policy)
                values.append(value)

            ss_index = 0 if index - self.config.num_observations < 0 else index - self.config.num_observations
            observations = states[ss_index:index]
            while len(observations) < self.config.num_observations:
                observations.insert(0, [0 for _ in range(self.config.observation_space_size)])
            target_rewards = rewards[index:index+self.config.num_unroll_steps+1]
            target_actions = actions[index:index+self.config.num_unroll_steps]
            target_policys = policys[index:index+self.config.num_unroll_steps]
            target_values = []
            for i in range(len(values[index:index+self.config.num_unroll_steps])):
                if i + self.config.td_steps >= len(values[index:]):
                    value = 0
                else:
                    value = values[index:][i+self.config.td_steps] * (self.config.discount ** self.config.td_steps)
                for j in range(self.config.td_steps):
                    if i + j + self.config.td_steps - 1 > len(rewards[index:]):
                        value += 0
                    else:
                        value += rewards[index:][i+j] * (self.config.discount ** j)
                target_values.append(value)

            samples.append((observations, (target_actions, target_policys, target_rewards[1:], target_values)))

        return samples, sample_size


class Trainer:
    def __init__(self, config):
        self.config = config
        if config.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam
        elif config.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD
        else:
            raise NotImplementedError(
                'Please select "Adam" or "SGD" in settings.py'
            )
        self.optimizer = optimizer(config.learning_rate)

    def train(self, model, replay_buffer):
        trainable_variables = model.representation.trainable_variables +\
                              model.dynamics.trainable_variables +\
                              model.prediction.trainable_variables
        samples, sample_size = replay_buffer.sample()
        with tf.GradientTape(persistent=True) as tape:
            all_losses = 0
            for data in samples:
                losses = 0
                hidden_state = model.representation(np.array([data[0]]))
                for i in range(self.config.num_unroll_steps):
                    policy_prob, value_prob = model.prediction(hidden_state)
                    action_true = data[1][0][i]
                    policy_target = data[1][1][i]
                    value_target = data[1][3][i]
                    reward_target = data[1][2][i]
                    hidden_state, reward_prob = model.dynamics([
                        hidden_state,
                        action_true
                    ])
                    losses += tf.keras.losses.mean_squared_error(
                        y_pred=policy_prob,
                        y_true=policy_target
                    ) + tf.keras.losses.mean_squared_error(
                        y_pred=value_prob,
                        y_true=np.array([[value_target]])
                    ) + tf.keras.losses.mean_squared_error(
                        y_pred=reward_prob,
                        y_true=np.array([[reward_target]])
                    )
                losses = losses/self.config.num_unroll_steps
                all_losses += losses

            all_losses = all_losses/sample_size
            grad = tape.gradient(all_losses, trainable_variables)
            self.optimizer.apply_gradients(zip(grad, trainable_variables))
            print(all_losses)