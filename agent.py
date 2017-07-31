import numpy as np

class Agent(object):
    def __init__(self, n, prior, policy, alpha=None):
        self.t = 0
        self.prior = prior
        self.n = n
        self.value_estimates = prior*np.ones(n)
        self.action_attempts = np.zeros(n)
        self.policy = policy
        self.alpha = alpha

    def reset(self):
        self.t = 0
        self.value_estimates = self.prior*np.ones(self.n)
        self.action_attempts = np.zeros(self.n)

    def choose(self):
        action = self.policy.choose(self)
        self.action_attempts[action] += 1
        self.last_action = action
        return action

    def observe(self, real_reward):
        self.t += 1

        if self.alpha is None:
            alpha = 1.0 / self.action_attempts[self.last_action]
        else:
            alpha = self.alpha

        self.value_estimates[self.last_action] += alpha*(real_reward - self.value_estimates[self.last_action])
