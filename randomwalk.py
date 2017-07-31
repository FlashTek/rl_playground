import numpy as np
from matplotlib import pyplot as plt

class TdChain(object):
    state = 0

    def __init__(self, gamma, n_states=7, alpha=0.1, values=None, prior=0.5, batch=False):
        self.gamma = gamma
        self.n_states = n_states
        self.state = self.n_states // 2
        self.terminated = False
        self.alpha = alpha
        self.batch = batch
        self.history = []

        if values is None:
            values = np.ones(n_states)*prior

        self.values = values

    def get_reward(self, state):
        if state == self.n_states-1:
            return 1.0
        else:
            return 0.0

    def update_value(self, state, new_state):
        reward = self.get_reward(new_state)
        if not (new_state == 0 or new_state == self.n_states-1):
            self.values[state] += self.alpha * (reward + self.gamma * self.values[new_state] - self.values[state])
        else:
            self.values[state] += self.alpha * (reward - self.values[state])

    def move(self):
        old_state = self.state

        choice = np.random.randint(0,2)
        if choice == 0:
            #move left
            new_state = self.state-1
            #print("move left")

        elif choice == 1:
            #move right
            new_state = self.state+1
            #print("move right")

        if not self.batch:
            self.update_value(self.state, new_state)

        self.history += [(self.state, new_state)]

        self.state = new_state

        if self.state == 0 or self.state == self.n_states-1:
            self.terminated = True

            if self.batch:
                updates = np.zeros(self.n_states)

                for (state, new_state) in self.history:
                    reward = 1.0 if new_state == self.n_states-1 else 0.0

                    if not (new_state == 0 or new_state == self.n_states-1):
                        updates[state] += self.alpha * (reward + self.gamma * self.values[new_state] - self.values[state])
                    else:
                        updates[state] += self.alpha * (reward - self.values[state])

                self.values += updates



class McChain(object):
    state = 0

    def __init__(self, gamma, n_states=7, alpha=0.1, values=None, prior=0.5, batch=False):
        self.gamma = gamma
        self.n_states = n_states
        self.state = self.n_states // 2
        self.terminated = False
        self.alpha = alpha
        self.batch = batch

        if values is None:
            values = np.ones(n_states)*prior

        self.values = values

        self.history = []

    def get_reward(self, state):
        if state == self.n_states-1:
            return 1.0
        else:
            return 0.0

    def move(self):
        old_state = self.state

        choice = np.random.randint(0,2)
        if choice == 0:
            #move left
            new_state = self.state-1
            #print("move left")

        elif choice == 1:
            #move right
            new_state = self.state+1
            #print("move right")

        self.state = new_state
        self.history += [self.state]

        if self.state == 0 or self.state == self.n_states-1:
            self.terminated = True

            reward = 1.0 if self.state == self.n_states-1 else 0.0

            if self.batch == True:
                updates = np.zeros(self.n_states)

                for state in self.history:
                    updates[state] += self.alpha*(reward - self.values[state])

                self.values += updates

            else:
                for state in self.history:
                    self.values[state] += self.alpha*(reward - self.values[state])

def main():
    n_episodes = 10000
    n_states = 7
    rms_values = []

    for i in range(4):
        values = np.ones(n_states)*0.5
        history = []
        true_values = np.array([1/6, 2/6, 3/6, 4/6, 5/6]).reshape((1, 5))
        for ep in range(n_episodes):
            if i == 0:
                chain = McChain(gamma=1.0, alpha=.001, values=values, n_states=n_states)
            elif i == 1:
                chain = McChain(gamma=1.0, alpha=.001, values=values, n_states=n_states, batch=True)
            elif i == 2:
                chain = TdChain(gamma=1.0, alpha=.001, values=values, n_states=n_states)
            elif i == 3:
                chain = TdChain(gamma=1.0, alpha=.001, values=values, n_states=n_states, batch=True)

            while not chain.terminated:
                chain.move()

            values = chain.values
            history += [values.copy()[1:n_states-1]]

        history = np.array(history)

        if i == 3:
            plt.plot(history)
            plt.show()

        history -= true_values
        rms = np.mean(np.square(history), axis=1)

        rms_values += [rms]

    rms_values = np.array(rms_values).T
    print(rms_values.shape)
    plt.plot(rms_values)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
