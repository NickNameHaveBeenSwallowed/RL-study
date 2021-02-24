from config import MuzeroConfig
from networks.full_connect import MuZeroNetwork

settings = {
    'game': 'CartPole-v1',
    'network': MuZeroNetwork,
    'num_simulations': 20,
    'action_space_size': 2,
    'observation_space_size': 4,
    'num_observations': 2,
    'is_zero_sum_game': False,
    'window_size': int(1e6),
    'training_steps': int(1e6),
    'optimizer': "Adam",
    'learning_rate': 5e-2,
    'learning_rate_decay_steps': 350e3,
    'learning_rate_decay_rate': 0.1,
    'batch_size': 128,
    'l2_regularization': 1e-4,
    'num_actors': 1,
    'num_unroll_steps': 5,
    'td_steps': 3,
    'value_loss_discount': 0.25,
    'dirichlet_alpha': 0.25,
    'discrete_support_size': 11,
    'discrete_floor': -5,
    'discrete_ceiling': 5,
    'visit_softmax_temperature_fn': 1.0,
    'discount': 0.997,
    'pb_c_init': 1.25,
    'pb_c_base': 19652
}

config = MuzeroConfig(**settings)