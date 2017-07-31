import numpy as np
import progressbar

class Environment(object):
    def __init__(self, bandit, agents):
        self.bandit = bandit
        self.agents = agents

    def reset(self):
        self.bandit.reset()

        for a in self.agents:
            a.reset()

    def run(self, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents), experiments))
        optimals = np.zeros_like(scores)

        bar = progressbar.ProgressBar(max_value=trials*experiments, redirect_stdout=True)

        for e in range(experiments):
            self.reset()

            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    real_reward, is_optimal = self.bandit.pull(action)
                    agent.observe(real_reward)

                    scores[t, i, e] = real_reward

                    if is_optimal:
                        optimals[t, i, e] = 1

                    bar.update(e*trials+t)

        bar.finish()

        mean_scores = np.empty((trials, len(self.agents)))
        mean_optimals = np.empty_like(mean_scores)
        std_scores = np.empty_like(mean_scores)
        std_optimals = np.empty_like(mean_scores)

        for t in range(trials):
            for i in range(len(self.agents)):
                mean_scores[t, i] = np.mean(scores[t, i])
                mean_optimals[t, i] = np.mean(optimals[t, i])
                std_scores[t, i] = np.std(scores[t, i])
                std_optimals[t, i] = np.std(optimals[t, i])

        return mean_scores, std_scores, mean_optimals, std_optimals
