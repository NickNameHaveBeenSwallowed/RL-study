import tensorflow as tf
import numpy as np

class MuZeroNetwork(object):
    def __init__(self, obs_shape: 'int', act_shape: 'int', discrete_support_size: 'int', obs_num=3, l2=1e-4):

        input_ops = tf.keras.Input(shape=(obs_num, obs_shape))
        x = tf.keras.layers.Flatten()(input_ops)
        x = tf.keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(x)
        x = tf.keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(x)
        hidden_state = tf.keras.layers.Dense(
            units=obs_shape,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            activation='tanh'
        )(x)
        self.representation = tf.keras.Model(inputs=input_ops, outputs=hidden_state)

        input_action = tf.keras.Input(shape=(act_shape))
        input_hidden_state = tf.keras.Input(shape=(obs_shape))
        hidden_state_embedding = tf.keras.layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(input_hidden_state)
        action_embedding = tf.keras.layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(input_action)

        x = tf.keras.layers.concatenate([hidden_state_embedding, action_embedding])

        x = tf.keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(x)
        next_hidden_state = tf.keras.layers.Dense(
            units=obs_shape,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            activation='tanh'
        )(x)
        reward = tf.keras.layers.Dense(
            units=discrete_support_size,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            activation='softmax'
        )(x)
        self.dynamics = tf.keras.Model(inputs=[input_hidden_state, input_action], outputs=[next_hidden_state, reward])

        input_hidden_state = tf.keras.Input(shape=(obs_shape))
        x = tf.keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(input_hidden_state)
        x = tf.keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(x)
        policy = tf.keras.layers.Dense(
            units=act_shape,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(x)
        value = tf.keras.layers.Dense(
            units=discrete_support_size,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        )(x)
        self.prediction = tf.keras.Model(inputs=input_hidden_state, outputs=[policy, value])

    def save_weights(self, save_path):
        self.representation.save_weights(save_path+'/representation.h5')
        self.dynamics.save_weights(save_path+'/dynamics.h5')
        self.prediction.save_weights(save_path+'/prediction.h5')

    def load_weights(self, load_path):
        self.representation.load_weights(load_path+'/representation.h5')
        self.dynamics.load_weights(load_path+'/dynamics.h5')
        self.prediction.load_weights(load_path+'/prediction.h5')
