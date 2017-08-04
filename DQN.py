# -*- coding: utf-8 -*-
#improves the output of keras on Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

import gym

import os.path

import sys
import codecs

import random

from keras.layers import Conv2D, Dense, Flatten, ConvLSTM2D, Input, Activation, BatchNormalization, Dropout
from keras.models import Model
from collections import deque
from keras.models import load_model
from keras.optimizers import Adam, RMSprop

from keras import regularizers
import keras.backend as K

class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return "epsilon greedy policy"

    def choose(self, agent):
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(agent.action_space.n)
        else:
            action_values = agent.evaluate_actions()
            choice = np.argmax(action_values)

            max_choices = np.where(action_values == action_values[choice])[0]

            if len(max_choices) > 1:
                choice = np.random.choice(max_choices)

        return choice

class EpsilonGreedyDecayPolicy(Policy):
    def __init__(self, initial_epsilon, final_epsilon, annealing_steps=1000000, current_step=0):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.annealing_steps = annealing_steps
        self.current_step = current_step

        self.state = (-1,-1)

    def __str__(self):
        return "dec.eps. policy: eps={0:.4f}, la: {1}".format(self.epsilon, self.state)

    def choose(self, agent):
        if np.random.rand() < self.epsilon:
            choice = np.random.randint(0, agent.action_space.n)
            self.state = (0, choice)
        else:
            action_values = agent.evaluate_actions()
            choice = np.argmax(action_values)

            max_choices = np.where(action_values == action_values[choice])[0]

            if len(max_choices) > 1:
                choice = np.random.choice(max_choices)

            self.state = (1, choice)

        if self.current_step < self.annealing_steps:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon)/self.annealing_steps

        self.current_step += 1

        return choice

class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return "greedy policy"

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [None for i in range(capacity)]
        self.position = 0
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return [self.data[item] for item in key]

    def __setitem(self, key, value):
        raise NotImplemented("Setting items by index is not allowed.")

    def append(self, x):
        if self.length < self.capacity:
            self.length += 1

        self.data[self.position] = x
        self.position = (self.position+1) % self.capacity

def huber(y_true, y_pred):
    error = y_true-y_pred
    return K.mean(K.sqrt(K.square(error) + 1.0) - 1.0, axis=-1)

class Agent(object):
    def __init__(self, environment, nb_frames=1, policy=GreedyPolicy(), Q_model=None, P_model=None, buffer_capacity=100,
                 idle_action=1, batch_size=32, gamma=1.0, observation_time=50000, P_update_interval=1000):
        self.environment = environment
        self.nb_frames = nb_frames
        self.action_space = self.environment.action_space

        self.image_rows = 84#environment.observation_space.shape[0]
        self.image_columns = 84#environment.observation_space.shape[1]

        self.gamma = gamma

        self.policy = policy
        self.observation = self.environment.reset()

        self.last_frames = np.empty((self.image_rows, self.image_columns, nb_frames))
        self.D = Memory(capacity=buffer_capacity)

        if Q_model is None:
            Q_model = self.build_model()
        if P_model is None:
            P_model = self.build_model()

        self.Q_model = Q_model
        self.P_model = P_model

        self.P_update_interval = P_update_interval

        self.idle_action = idle_action
        self.batch_size = batch_size

        self.observation_time = observation_time

        self.action = self.idle_action

        self._total_time = 0

        self.reset()

    def build_model(self):
        input_layer = Input(shape=(self.image_rows, self.image_columns, self.nb_frames))
        """
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='same')(input_layer)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        """
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='same')(input_layer) #, kernel_regularizer=regularizers.l2(0.01)
        #x = BatchNormalization()(x)
        x = Activation("relu")(x)
        #x = Dropout(0.1)(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation("relu")(x)
        #x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation("relu")(x)
        x = Dense(self.action_space.n, activation="linear")(x)

        model = Model(inputs=[input_layer], outputs=[x])
        #model.compile(optimizer="adam", loss="mse")
        opt = Adam(lr=0.00025)#RMSprop()#lr=0.00025, epsilon=0.01)
        model.compile(optimizer=opt, loss=huber)

        return model

    def observe(self, action):
        observation, reward, done, _ = self.environment.step(action)
        observation = ((resize(rgb2gray(observation), (self.image_rows, self.image_columns), mode="constant"))-0.5)*2.0
        self.observation = observation
        self.last_frames[:, :, :3] = self.last_frames[:, :, 1:]
        self.last_frames[:, :, 3] = self.observation

        return observation, reward, done

    def reset(self):
        self._time = 0
        self._done = False
        self.loss = 0
        self.cumloss = 0
        self.state = self.last_frames.copy()
        self.cumreward = 0

    def idle_observe(self):
        for t in range(self.observation_time):
            if t%500==0:
                print("\r{0:.1f}%".format(t/self.observation_time*100), end="")

            _, reward, done = self.observe(self.idle_action)

            if t == self.nb_frames:
                self.state = self.build_state()
            elif t > self.nb_frames:
                if t % self.nb_frames == 0:
                    old_state = self.state.copy()
                    self.state = self.build_state()
                    self.D.append([old_state, self.action, reward, self.state, done])

            if done:
                self.environment.reset()

        self.state = self.build_state()

    def choose_action(self):
        return self.policy.choose(self)

    def print_status(self):
        print("{0} \t R={0} \t L={1:8f} \t CumL={2:8f}".format(self.policy, self.cumreward, self.loss, self.cumloss))

    def build_state(self):
        return self.last_frames.copy()

    #@profile
    def step(self):
        if self.time > 0 and self.time % self.nb_frames == 0:
            self.action = self.choose_action()

        old_state = self.state.copy()

        _, reward, done = self.observe(self.action)

        self.cumreward += reward

        self.D.append([old_state, self.action, reward, self.state, done])

        if len(self.D) > self.batch_size:
            minibatch = self.D[random.sample(range(len(self.D)), self.batch_size)]

            states = np.empty((self.batch_size, self.image_rows, self.image_columns, self.nb_frames))
            new_states = np.empty((self.batch_size, self.image_rows, self.image_columns, self.nb_frames))
            targets = np.zeros((self.batch_size, self.action_space.n))
            rewards = np.zeros((self.batch_size, 1))
            actions = np.zeros((self.batch_size, 1), dtype=int)
            terminals = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                states[i], actions[i], rewards[i], new_states[i], terminals[i] = minibatch[i]

            Q_pred = self.P_model.predict(new_states)

            targets = self.Q_model.predict(states)
            targets[np.arange(self.batch_size), actions] = rewards + (1-terminals)*self.gamma*np.max(Q_pred, axis=1)

            self.loss = self.Q_model.train_on_batch(states, targets)
            self.cumloss += self.loss

        if self._total_time % self.P_update_interval == 0:
            [self.P_model.trainable_weights[i].assign(self.Q_model.trainable_weights[i]) for i in range(len(self.Q_model.trainable_weights))]

        self._done = done
        self._time += 1

    def evaluate_actions(self):
        self.state = self.build_state()

        return self.Q_model.predict(np.array([self.state]))[0]

    @property
    def time(self):
        return self._time

    @property
    def done(self):
        return self._done


def main():
    """
    import glob
    import h5py

    model_files = sorted(glob.glob('*.h5'))
    for model_file in model_files:
        print("Update '{}'".format(model_file))
        with h5py.File(model_file, 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']
    """

    env_name = "SpaceInvaders-v0"
    nb_episodes = 200

    agent_nb_frames = 4

    env = gym.make(env_name)
    env.reset()

    model = None
    if os.path.isfile("DQN_model.h5"):
        model = load_model("DQN_model.h5", custom_objects={"huber": huber})
        print("model loaded.")

    #policy=EpsilonGreedyPolicy(epsilon=0.1),
    agent = Agent(env, nb_frames=agent_nb_frames, batch_size=32,
                  policy=EpsilonGreedyDecayPolicy(initial_epsilon=1.0, final_epsilon=0.10, annealing_steps=5000, current_step=1),
                  #policy=EpsilonGreedyDecayPolicy(initial_epsilon=0.05, final_epsilon=0.05, annealing_steps=1000, current_step=1000),
                  buffer_capacity=50000, observation_time=10000, gamma=0.99, Q_model=model, P_model=model, idle_action=0)

    print("starting idle observation...")
    agent.idle_observe()
    print("finished idle observation.")

    for n in range(nb_episodes):
        env.reset()
        agent.reset()

        while not agent.done:
            env.render()
            agent.step()

            if agent.time % agent_nb_frames == 0:
                print(n, end="\t")
                agent.print_status()

        print("---episode finished---")

    agent.model.save("DQN_model.h5", overwrite=True)

if __name__ == "__main__":
    main()
