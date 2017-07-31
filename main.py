import numpy as np

from policy import *
from agent import *
from environment import *
from bandit import *

from matplotlib import pyplot as plt

def main():
    print("setting up bandits...")
    n_bandits = 10
    mean = 0.0
    variance = 1.0

    #bandit = GaussianBandit(k=n_bandits, mu=mean, sigma=variance)
    bandit = RandomWalkGaussianBandit(k=n_bandits, mu=mean, sigma=variance, random_range=10.0)
    agents = []

    agentSettings = [(0, (0.0, 0.1, 0.1)), (0, (0.0, 0.1, None))]#[(1, 0.0), (1, 5.0), (0, (0.0, 0.0)), (0, (0.0, 0.01)), (0, (0.0, 0.1)), (0, (5.0, 0.0)), (0, (5.0, 0.01)), (0, (5.0, 0.1))]

    for typ, data in agentSettings:
        if typ == 0:
            init, eps, alpha = data
            epsGreedyPolicy = EpsilonGreedyPolicy(eps)
            agent = Agent(n_bandits, init, epsGreedyPolicy, alpha)
        elif typ == 1:
            init = data
            softmaxPolicy = SoftmaxPolicy(init_temperature=10)
            agent = Agent(n_bandits, init, softmaxPolicy)
        agents += [agent]



    env = Environment(bandit, agents)

    m_sc, s_sc, m_opt, s_opt = env.run(trials=2000, experiments=500)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i in range(len(agentSettings)):
        #ax1.errorbar(np.arange(500), m_sc[:, i], s_sc[:, i], label=agentSettings[i])
        #ax2.errorbar(np.arange(500), m_opt[:, i], s_opt[:, i], label=agentSettings[i])

        ax1.plot(m_sc[:, i], label=agentSettings[i])
        ax2.plot(m_opt[:, i], label=agentSettings[i])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=False, shadow=False, ncol=len(agentSettings)//2)
    plt.show()


if __name__ == '__main__':
    main()
