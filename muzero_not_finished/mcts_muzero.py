import numpy as np

class MinMax:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class TreeNode:
    def __init__(
            self,
            parent,
            prior_p,
            hidden_state,
            reward,
            is_zero_sum_game,
            discount=0.997
    ):
        self._parent = parent
        self._children = {}
        self._num_visits = 0
        self.value = 0
        self._U = 0
        self._P = prior_p

        self._hidden_state = hidden_state
        self.reward = reward

        self._is_zero_sum_game = is_zero_sum_game
        self._discount = discount

    def expand(self, action_priorP_hiddenStates_reward):
        for action, prob, hidden_state, reward in action_priorP_hiddenStates_reward:
            if action not in self._children.keys():
                self._children[action] = TreeNode(
                    parent=self,
                    prior_p=prob,
                    hidden_state=hidden_state,
                    reward=reward,
                    is_zero_sum_game=self._is_zero_sum_game,
                    discount=self._discount
                )

    def select(self, minmax, pb_c_init=1.25, pb_c_base=19652):
        return max(
            self._children.items(),
            key=lambda node_tuple: node_tuple[1].get_value(minmax, pb_c_init, pb_c_base)
        )

    def _update(self, reward, value, minmax):
        _G = reward + value
        self.value = (self._num_visits * self.value + _G) / (self._num_visits + 1)
        minmax.update(self.value)
        self._num_visits += 1

    def backward_update(self, minmax, value, backward_reward=0):
        self._update(
            reward=backward_reward,
            value=value,
            minmax=minmax
        )
        if self._parent:
            if self._is_zero_sum_game:
                self._parent.backward_update(
                    minmax=minmax,
                    value=self._discount * (-value),
                )
            else:
                self._parent.backward_update(
                    minmax=minmax,
                    value=self._discount * value,
                    backward_reward=self.reward + self._discount * backward_reward
                )

    def get_value(self, minmax, pb_c_init=1.25, pb_c_base=19652):
        self._U = self._P *\
                  (np.sqrt(self._parent._num_visits)/(1 + self._num_visits)) *\
                  (
                    pb_c_init + np.log(
                      (self._parent._num_visits + pb_c_base + 1)/pb_c_base)
                  )

        return minmax.normalize(self.value) + self._U

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def __str__(self):
        return "TreeNode"

class MCTS:
    def __init__(
            self,
            train,
            model,
            observations,
            action_num,
            root_dirichlet_alpha,
            root_exploration_fraction,
            is_zero_sum_game,

            discount=0.997,
            discrete_support_size=601,
            discrete_floor=-300,
            discrete_ceiling=300,

            num_simulations=50,
            pb_c_init=1.25,
            pb_c_base=19652,
    ):
        self.train = train
        self._muzero_model = model
        self.action_num = action_num
        self._minmax = MinMax()
        self.root = TreeNode(
            parent=None,
            prior_p=0,
            hidden_state=self._muzero_model.representation(observations),
            reward=None,
            is_zero_sum_game=is_zero_sum_game,
            discount=discount
        )
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction

        self.discrete_support_size = discrete_support_size
        self.discrete_floor = discrete_floor
        self.discrete_ceiling = discrete_ceiling

        self._num_simulations = num_simulations
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base

    def add_exploration_noise(self, node):
        actions = list(node._children.keys())
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(actions))
        frac = self.root_exploration_fraction
        for a, n in zip(actions, noise):
            node._children[a]._P = node._children[a]._P * (1 - frac) + n * frac

    def _simulations(self):
        node = self.root
        while True:
            if node.is_leaf():
                break
            _, node = node.select(self._minmax, self.pb_c_init, self.pb_c_base)

        action_probs, value = self._muzero_model.prediction(node._hidden_state)
        action_probs = action_probs[0]

        value = self._muzero_model.inverse_discrete_transform(value, self.discrete_support_size, self.discrete_ceiling)[0]

        action_priorP_hiddenStates_reward = []

        action_index = 0

        for action_prob in action_probs:
            action = action_index

            action_one_hot = [1 if i == action_index else 0 for i in range(self.action_num)]

            action_index += 1

            next_hidden_state, reward = self._muzero_model.dynamics([
                node._hidden_state,
                np.array([action_one_hot])
            ])
            reward = self._muzero_model.inverse_discrete_transform(reward, self.discrete_support_size, self.discrete_ceiling)[0]

            action_priorP_hiddenStates_reward.append((action, action_prob, next_hidden_state, reward))

        node.expand(action_priorP_hiddenStates_reward)

        if self.train:
            self.add_exploration_noise(node)

        node.backward_update(minmax=self._minmax, value=value)

    def get_action_prob(self, T=1.0):

        for _ in range(self._num_simulations + 1):
            self._simulations()

        actions = []
        visits = []
        for action, node in self.root._children.items():
            actions.append(action)
            visits.append(node._num_visits)

        return actions, np.array(visits) ** (1 / T) / np.sum(np.array(visits) ** (1 / T))

    def __str__(self):
        return "MCTS"