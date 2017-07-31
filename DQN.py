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

from keras.layers import Conv2D, Dense, Flatten, ConvLSTM2D, Input, Activation
from keras.models import Model
from collections import deque
from keras.models import load_model
from keras.optimizers import Adam, RMSprop

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

    def __str__(self):
        return "dec.eps. policy: eps={0:.4f}".format(self.epsilon)

    def choose(self, agent):
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(agent.action_space.n)
        else:
            action_values = agent.evaluate_actions()
            choice = np.argmax(action_values)

            max_choices = np.where(action_values == action_values[choice])[0]

            if len(max_choices) > 1:
                choice = np.random.choice(max_choices)

        if self.current_step < self.annealing_steps:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon)/self.annealing_steps

        self.current_step += 1

        return choice

class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return "greedy policy"

class Agent(object):
    def __init__(self, environment, nb_frames=1, policy=GreedyPolicy(), model=None, buffer_capacity=100,
                 idle_action=1, batch_size=32, gamma=1.0, observation_time=50000):
        self.environment = environment
        self.nb_frames = nb_frames
        self.action_space = self.environment.action_space

        self.image_rows = 84#environment.observation_space.shape[0]
        self.image_columns = 84#environment.observation_space.shape[1]

        self.gamma = gamma

        self.policy = policy
        self.observation = self.environment.reset()

        self.frame_buffer = deque()
        self.D = deque()

        self.frame_buffer_capacity = buffer_capacity

        if model is None:
            model = self.build_model()
        self.model = model

        self.idle_action = idle_action
        self.batch_size = batch_size

        self.observation_time = observation_time

        self.action = self.idle_action

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
        x = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')(input_layer)
        x = Activation("relu")(x)
        x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = Dense(self.action_space.n)(x)

        model = Model(inputs=[input_layer], outputs=[x])
        #model.compile(optimizer="adam", loss="mse")
        opt = RMSprop(lr=0.00025, epsilon=0.01)
        model.compile(optimizer=opt, loss="mse")

        return model

    def observe(self, action):
        observation, reward, done, _ = self.environment.step(action)
        observation = (resize(rgb2gray(observation), (self.image_rows, self.image_columns), mode="constant")  - 127.0)/127.0
        self.observation = observation
        self.frame_buffer.append(self.observation)

        return observation, reward, done

    def reset(self):
        self._time = 0
        self._done = False
        self.loss = 0
        self.cumloss = 0
        self.state = None

        if len(self.frame_buffer) > 0:
            self.state = self.build_state()

    def idle_observe(self):
        for t in range(self.observation_time):
            if t%100==0:
                print(t/self.observation_time)

            _, reward, done = self.observe(self.idle_action)

            if t == self.nb_frames:
                self.state = self.build_state()
            if t > self.nb_frames:
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
        print("{0} \t L={1} \t CumL={2}".format(self.policy, self.loss, self.cumloss))

    def build_state(self):
        state = np.empty((self.image_rows, self.image_columns, self.nb_frames))

        for i in range(self.nb_frames):
            state[:, :, i] = self.frame_buffer[-i]

        return state

    #@profile
    def step(self):
        if self.time > 0 and self.time % self.nb_frames == 0:
            self.action = self.choose_action()

        old_state = self.state.copy()

        _, reward, done = self.observe(self.action)

        self.D.append([old_state, self.action, reward, self.state, done])

        if len(self.D) > self.batch_size:
            minibatch = random.sample(self.D, self.batch_size)

            states = np.empty((self.batch_size, self.image_rows, self.image_columns, self.nb_frames))
            new_states = np.empty((self.batch_size, self.image_rows, self.image_columns, self.nb_frames))
            targets = np.zeros((self.batch_size, self.action_space.n))
            rewards = np.zeros((self.batch_size, 1))
            actions = np.zeros((self.batch_size, 1), dtype=int)
            terminals = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                states[i], actions[i], rewards[i], new_states[i], terminals[i] = minibatch[i]

            #targets = self.model.predict(states)
            Q_sa = self.model.predict(new_states)

            targets[np.arange(self.batch_size), actions] = rewards + (1-terminals)*self.gamma*np.max(Q_sa, axis=1)

            """
            for i in range(self.batch_size):
                if terminals[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma*np.max(Q_sa[i])
            """
            self.loss = self.model.train_on_batch(states, targets)
            self.cumloss += self.loss

        if len(self.frame_buffer) > self.frame_buffer_capacity:
            self.frame_buffer.popleft()
            self.D.popleft()

        self._done = done
        self._time += 1

    def evaluate_actions(self):
        self.state = self.build_state()

        return self.model.predict(np.array([self.state]))[0]

    @property
    def time(self):
        return self._time

    @property
    def done(self):
        return self._done


def main():
    env_name = "Breakout-v0"
    nb_episodes = 500

    agent_nb_frames = 4

    env = gym.make(env_name)
    env.reset()

    model = None
    if os.path.isfile("DQN_model.h5"):
        model = load_model("DQN_model.h5")
        print("model loaded.")

    #policy=EpsilonGreedyPolicy(epsilon=0.1),
    agent = Agent(env, nb_frames=agent_nb_frames,
                  policy=EpsilonGreedyDecayPolicy(initial_epsilon=0.2, final_epsilon=0.05, annealing_steps=10000, current_step=0),
                  #policy=EpsilonGreedyDecayPolicy(initial_epsilon=0.05, final_epsilon=0.05, annealing_steps=1000, current_step=1000),
                  buffer_capacity=1000000, observation_time=10000, gamma=0.99, model=model, idle_action=1)

    print("starting idle observation...")
    agent.idle_observe()
    print("finished idle observation of {0} steps.".format(len(agent.frame_buffer)))

    for n in range(nb_episodes):
        env.reset()
        agent.reset()

        while not agent.done:
            env.render()
            agent.step()

            if agent.time % 10 == 0:
                print(n, end="\t")
                agent.print_status()

        print("---episode finished---")

    agent.model.save("DQN_model.h5", overwrite=True)

if __name__ == "__main__":
    main()
