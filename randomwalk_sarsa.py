import numpy as np
from matplotlib import pyplot as plt

class SarsaChain(object):
    state = 0

    def __init__(self, gamma, n_states=7, alpha=0.1, action_state_values=None, prior=0.5, epsilon=.01):
        self.gamma = gamma
        self.n_states = n_states
        self.state = self.n_states // 2
        self.terminated = False
        self.alpha = alpha
        self.epsilon = epsilon

        if action_state_values is None:
            action_state_values = np.ones(n_states, 2) * prior
            action_state_values[0, :] = 0
            action_state_values[-1, :] = 0

        self.action_state_values = action_state_values

        self.action = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            choice = np.random.randint(0, 2)
        else:
            choice = np.argmax(self.action_state_values[state])

            max_choices = np.where(self.action_state_values[state] == self.action_state_values[state][choice])[0]

            if len(max_choices) > 1:
                choice = np.random.choice(max_choices)
        #print(np.argmax(self.action_state_values[state]))
        #print(choice)
        #return selected action
        return choice

    def get_reward(self, state):
        if state == self.n_states-1:
            return 1.0
        else:
            return 0.0


    def update_value(self, state, new_state, action):
        reward = self.get_reward(new_state)
        new_action = self.choose_action(new_state)

        self.action_state_values[state, action] += self.alpha * (reward + self.gamma*self.action_state_values[new_state, new_action] - self.action_state_values[state, action])

        self.action = new_action
        self.state = new_state

    def move(self):
        if self.action == 0:
            #move left
            new_state = self.state-1
            #print("move left")

        elif self.action == 1:
            #move right
            new_state = self.state+1
            #print("move right")

        self.update_value(self.state, new_state, self.action)

        if self.state == 0 or self.state == self.n_states-1:
            self.terminated = True

def main():
    n_episodes = 1000
    n_states = 20
    history = []

    action_state_values = np.ones((n_states, 2))*0.5
    action_state_values[0, :] = 0
    action_state_values[n_states-1, :] = 0

    suc, fail = 0,0

    rates = []

    for n in range(n_episodes):
        if n % 100 == 0:
            print(n)
        chain = SarsaChain(gamma=1.0, alpha=.1  , action_state_values=action_state_values, n_states=n_states, epsilon=.1)

        while not chain.terminated:
            chain.move()

        if chain.state == 0:
            fail +=1
        else:
            suc += 1
        rates += [[suc, fail]]


        action_state_values = chain.action_state_values.copy()
        history += [action_state_values.copy()]

    history = np.array(history)

    print(history.shape)

    plt.plot(np.array(rates))
    plt.show()

    fig, ax = plt.subplots(2, 1)
    for i in range(n_states):
        ax[0].plot(history[:, i, 0], label=i)

        ax[1].plot(history[:, i, 1], label=i)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
