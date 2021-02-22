from mcts_muzero import MCTS
from collections import deque
import numpy as np

class MuZeroActor:
    def __init__(
            self,
            config,
            is_training
    ):
        self.config = config
        self.model = config.build_model()
        self.T = config.visit_softmax_temperature_fn
        self.initialize()
        self.train = is_training

    def step(self, state):

        self.observations_queue.append(state)

        mcts = MCTS(
            train=self.train,
            model=self.model,
            observations=np.array([self.observations_queue]),
            action_num=self.config.action_space_size,
            root_dirichlet_alpha=self.config.root_dirichlet_alpha,
            root_exploration_fraction=self.config.root_exploration_fraction,
            is_zero_sum_game=self.config.is_zero_sum_game,
            discount=self.config.discount,
            num_simulations=self.config.num_simulations,
            pb_c_init=self.config.pb_c_init,
            pb_c_base=self.config.pb_c_base
        )

        actions, policy = mcts.get_action_prob(T=self.T)
        mean_value = np.array(mcts.root.value[0][0])

        return np.random.choice(actions, p=policy), np.array([policy]), mean_value

    def initialize(self):
        self.observations_queue = deque(maxlen=self.config.num_observations)

        for _ in range(self.config.num_observations):
            self.observations_queue.append([0 for __ in range(self.config.observation_space_size)])
