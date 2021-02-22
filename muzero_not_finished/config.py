class MuzeroConfig:
    def __init__(
            self,
            game,
            network,
            num_simulations,
            action_space_size,
            observation_space_size,
            num_observations,
            is_zero_sum_game,

            optimizer,
            learning_rate,
            learning_rate_decay_steps,
            learning_rate_decay_rate,
            batch_size,
            l2_regularization,
            num_actors,
            num_unroll_steps,
            td_steps,

            dirichlet_alpha,
            visit_softmax_temperature_fn=1.0,
            discount=0.997,

            pb_c_init=1.25,
            pb_c_base=19652,
    ):
        ## Game
        self.game = game
        self.network = network
        self.num_simulations = num_simulations
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.num_observations = num_observations
        self.is_zero_sum_game = is_zero_sum_game

        ## Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.batch_size = batch_size
        self.l2_regularization = l2_regularization
        self.num_actors = num_actors

        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps

        ## MCTS setting
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.discount = discount
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base

        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

    def build_model(self):
        return self.network(
            obs_shape=self.observation_space_size,
            act_shape=self.action_space_size,
            obs_num=self.num_observations,
            l2=self.l2_regularization
        )