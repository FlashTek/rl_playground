#improves the output of keras on Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import numpy as np
import keras as K
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import OneHotEncoder

def build_model(lr):
    state_input_layer = Input((2,))
    action_input_layer = Input((3,))
    merge = concatenate([state_input_layer, action_input_layer])

    x = Dense(25, activation="sigmoid")(merge)
    x = Dense(10, activation="sigmoid")(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="sigmoid")(x)
    x = Dropout(0.1)(x)
    # output_layer = Dense(1, activation="tanh")(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=[state_input_layer, action_input_layer], outputs=[output_layer])
    opt = SGD(lr=lr)
    model.compile("sgd", "mse")
    model.summary()
    return model

class MountainCar(object):
    def __init__(self, alpha=.001, gamma=0.9, epsilon=0.1, boundaries=(-1.2, 0.5), velocity_boundaries=(-0.07, 0.07), model=None):
        self.position = np.random.rand()*0.2-0.6
        self.velocity = 0.0

        self.boundaries = boundaries
        self.velocity_boundaries = velocity_boundaries

        self.actions = {-1: "move left", 0: "idle", 1: "move right"}
        self.action = 0
        self.is_terminal = False

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        if model is None:
            model = build_model(lr=alpha)
        self.model = model
        self.action_encoder = OneHotEncoder().fit(np.array([0, 1, 2]).reshape((-1, 1)))

        self.time = 0

    def process_action(self, action):
        if self.position < self.boundaries[0]:
            self.velocity = 0.0
        self.position = np.clip(self.position+self.velocity, *self.boundaries)
        self.velocity = np.clip(self.velocity + 0.001*action-0.0025*np.cos(3*self.position), *self.velocity_boundaries)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(-1, 2)
        else:
            state = np.array([self.position, self.velocity])

            actions = np.array([-1, 0, 1]).reshape(-1, 1)
            actions = self.encode_action(actions)

            q_prediction_input_state = np.tile(state, len(actions)).reshape((len(actions), -1))

            q_predictions = self.model.predict([q_prediction_input_state, actions])

            action = np.argmax(actions[np.argmax(q_predictions)])-1

        return action

    def encode_action(self, action):
        return self.action_encoder.transform(np.array([action]).reshape((-1, 1))+1).A

    def move(self):
        old_state = np.array([self.position, self.velocity]).reshape((-1, 2))

        self.process_action(self.action)
        new_state = np.array([self.position, self.velocity]).reshape((-1, 2))

        if self.position >= self.boundaries[1]:
            reward = 0
            prediction = [reward]

            self.is_terminal = True
            new_action = self.action
        else:
            new_action = self.choose_action()
            reward = -1

            new_action_encoded = self.encode_action(new_action)
            prediction = [reward + self.gamma*self.model.predict([new_state, new_action_encoded])[0]]

        prediction = np.array([prediction]).reshape((-1, 1))
        action_encoded = self.encode_action(self.action)

        fit_res = self.model.train_on_batch([old_state, action_encoded], prediction)

        self.time += 1

        self.action = new_action

def main():
    nb_episodes = 1000

    model = None

    steps = []
    for n in range(nb_episodes):
        car = MountainCar(model=model, epsilon=0.1)
        allowed_actions = car.encode_action(np.array([-1, 0, 1]).reshape(-1, 1))

        t = 0
        positions = []
        while not car.is_terminal:
            car.move()
            t+= 1
            positions += [car.position]

            if t%100 == 0:
                car_state = np.tile(np.array([car.position, car.velocity]).reshape((-1, 2)),
                                                   len(allowed_actions)).reshape((len(allowed_actions), -1))

                print("\t" + str(t), car.position, car.velocity, car.model.predict([car_state, allowed_actions]) )

        #plot the position curve
        print(t)
        steps += [t]

        model = car.model

main()
