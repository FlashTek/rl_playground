import numpy as np

class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True

class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)

class RandomWalkGaussianBandit(GaussianBandit):

    def __init__(self, k, mu=0, sigma=1, random_range=1.0):
        super(RandomWalkGaussianBandit, self).__init__(k, mu, sigma)
        self.random_range = random_range

    def pull(self, action):
        self.action_values += np.random.randint(-5,5, size=self.k)*0.01
        self.action_values[self.action_values < -self.random_range] = -self.random_range
        self.action_values[self.action_values > +self.random_range] = +self.random_range
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)
