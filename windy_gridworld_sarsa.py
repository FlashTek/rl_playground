import numpy as np
from matplotlib import pyplot as plt

class SarsaWindyGridworld(object):
    state = 0

    def __init__(self, gamma, n_rows, n_columns, alpha=0.1, action_state_values=None, prior=0.5, epsilon=.01, goal_state=(0,0), wind=None):
        self.gamma = gamma
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.state = (4, 0)
        self.terminated = False
        self.alpha = alpha
        self.epsilon = epsilon
        self.goal_state = goal_state

        if wind is None:
            wind = np.zeros((n_columns))

        self.wind = wind.astype(int)

        if action_state_values is None:
            action_state_values = np.ones((n_rows, n_colunmns, 4)) * prior
            action_state_values[self.goal_state, :] = 0

        self.action_state_values = action_state_values

        self.action = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            choice = np.random.randint(0, 4)
        else:
            choice = np.argmax(self.action_state_values[state])

            max_choices = np.where(self.action_state_values[state] == self.action_state_values[state][choice])[0]

            if len(max_choices) > 1:
                choice = np.random.choice(max_choices)

        return choice

    def get_reward(self, state):
        if state == self.goal_state:
            return 0.0#1.0
        elif 0 in state or state[0] == self.n_rows-1 or state[1] == self.n_columns-1:
            return -1.0
        else:
            return -1.0#0.0


    def update_value(self, state, new_state, action):
        reward = self.get_reward(new_state)
        new_action = self.choose_action(new_state)

        self.action_state_values[state[0], state[1], action] += self.alpha * (reward + self.gamma*self.action_state_values[new_state[0], new_state[1], new_action] - self.action_state_values[state[0], state[1], action])

        self.action = new_action
        self.state = new_state

    def move(self):
        if self.action == 0:
            new_state = tuple(np.subtract(self.state, (1, 0)))
        elif self.action == 1:
            new_state = tuple(np.add(self.state, (1, 0)))
        elif self.action == 2:
            new_state = tuple(np.subtract(self.state, (0, 1)))
        elif self.action == 3:
            new_state = tuple(np.add(self.state, (0, 1)))

        if (new_state[1] < self.n_columns and new_state[1] > 0):
            new_state = (new_state[0]+self.wind[new_state[1]], new_state[1])


        #do not move beyond the boundaries
        if new_state[0] < 0:
            new_state = (0, new_state[1])
        elif new_state[0] >= self.n_rows:
            new_state = (self.n_rows-1, new_state[1])
        elif new_state[1] < 0:
            new_state = (new_state[0], 0)
        elif new_state[1] >= self.n_columns:
            new_state = (new_state[0], self.n_columns-1)

        self.update_value(self.state, new_state, self.action)

        if self.state == self.goal_state:
            self.terminated = True

def main():
    n_episodes = 200
    n_rows = 7
    n_columns = 10
    goal_state = (3,7)
    history = []

    action_state_values = np.ones((n_rows, n_columns, 4))*0.1
    action_state_values[goal_state] *= 0

    wind = np.array([0,0,0,1,1,1,2,2,1,0])

    lengths = []

    for n in range(n_episodes):
        if n % 100 == 0:
            print(n)
        grid = SarsaWindyGridworld(gamma=1.0, alpha=.5 , action_state_values=action_state_values, n_rows=n_rows, n_columns=n_columns, epsilon=.1, goal_state=goal_state, wind=wind)

        moves = 0
        hist = []
        while not grid.terminated:
            grid.move()
            moves += 1

        lengths += [moves]

        action_state_values = grid.action_state_values.copy()
        history += [action_state_values.copy()]

    history = np.array(history)

    plt.plot(np.array(lengths), label="#moves")
    plt.legend()
    plt.show()

    move_history = []
    grid = SarsaWindyGridworld(gamma=1.0, alpha=.5, action_state_values=action_state_values, n_rows=n_rows, n_columns=n_columns, epsilon=0.0, goal_state=goal_state, wind=wind)

    while not grid.terminated:
        move_history += [grid.state]
        grid.move()
    move_history += [grid.state]

    move_history = np.array(move_history)
    print(move_history.shape)

    for i in range(1, move_history.shape[0]):
        plt.scatter(move_history[i, 1], move_history[i, 0], marker="$"+str(i)+"$", c=0)
    plt.show()

if __name__ == '__main__':
    main()
