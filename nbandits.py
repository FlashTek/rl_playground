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

    bandit = GaussianBandit(k=n_bandits, mu=mean, sigma=variance)
    agents = []

    agentSettings = [(0.0, 0.0), (0.0, 0.01), (0.0, 0.1), (5.0, 0.0), (5.0, 0.01), (5.0, 0.1)]

    for init, eps in agentSettings:
        epsGreedyPolicy = EpsilonGreedyPolicy(eps)
        epsGreedyAgent = Agent(n_bandits, init, epsGreedyPolicy)
        agents += [epsGreedyAgent]


    env = Environment(bandit, agents)

    sc, opt = env.run(trials=1000, experiments=1000)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i in range(len(agentSettings)):
        ax1.plot(sc[:, i], label=agentSettings[i])
        ax2.plot(opt[:, i], label=agentSettings[i])
    #ax1.legend(loc="lower right")
    #ax2.legend(loc="lower right")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=False, shadow=False, ncol=3)
    plt.show()

if __name__ == '__main__':
    main()
