import numpy as np

class TdAgent(object):
    def __init__(self, n_actions, n_states, prior, policy, alpha=None, gamma=1.0):
        self.t = 0
        self.prior = prior
        self.n_actions = n_actions
        self.n_states = n_states
        self.value_estimates = prior*np.ones(n_states)
        self.action_attempts = np.zeros(n_actions)
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma

        self.state = 0

        self.last_action = None
        self.new_state = None

    def reset_episode(self):
        self.t = 0
        self.state = 0

    def reset(self):
        self.reset_episode()
        
        self.value_estimates = self.prior*np.ones(self.n_states)
        self.action_attempts = np.zeros(self.n_actions)

    def choose(self):
        action, new_state = self.policy.choose(self, self.state)
        self.action_attempts[action] += 1
        self.last_action = action
        self.new_state = new_state
        return action

    def observe(self, real_reward):
        self.t += 1

        if self.alpha is None:
            alpha = 1.0 / self.t #self.action_attempts[self.last_action]
        else:
            alpha = self.alpha

        self.value_estimates[self.state] += alpha*(real_reward + self.gamma*self.value_estimates[self.new_state] - self.value_estimates[self.state])
        self.state = self.new_state
