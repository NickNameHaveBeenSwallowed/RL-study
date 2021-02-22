from config import MuzeroConfig
from networks.full_connect import MuZeroNetwork

settings = {
    'game': 'CartPole-v1',
    'network': MuZeroNetwork,
    'num_simulations': 20,
    'action_space_size': 2,
    'observation_space_size': 4,
    'num_observations': 3,
    'is_zero_sum_game': False,
    'optimizer': "Adam",
    'learning_rate': 5e-2,
    'learning_rate_decay_steps': 350e3,
    'learning_rate_decay_rate': 0.1,
    'batch_size': 128,
    'l2_regularization': 1e-2,
    'num_actors': 1,
    'num_unroll_steps': 5,
    'td_steps': 2,
    'dirichlet_alpha': 0.25,
    'visit_softmax_temperature_fn': 1.0,
    'discount': 0.997,
    'pb_c_init': 1.25,
    'pb_c_base': 19652
}

config = MuzeroConfig(**settings)