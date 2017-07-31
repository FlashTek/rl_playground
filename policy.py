import numpy as np

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
            choice = np.random.choice(len(agent.value_estimates))
        else:
            choice = np.argmax(agent.value_estimates)

            max_choices = np.where(agent.value_estimates == agent.value_estimates[choice])[0]

            if len(max_choices) > 1:
                choice = np.random.choice(max_choices)

        return choice

class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return "greedy policy"

class SoftmaxPolicy(Policy):
    def __init__(self, init_temperature):
        self.init_temperature = init_temperature

    def __str__(self):
        return "softmax policy"

    def choose(self, agent):
        temperature = self.init_temperature/(agent.t+1)

        probabilities = np.exp(agent.value_estimates/temperature)



        if (len(np.where(np.isnan(probabilities))[0]) > 0):
            print("value")
            isnan = np.isnan(probabilities)
            probabilities[:] = 0.0
            probabilities[isnan] = 1.0

        #print(probabilities)

        probabilities /= np.sum(probabilities)
        #print(probabilities)

        choice = np.random.choice(len(agent.value_estimates), p=probabilities)

        return choice
