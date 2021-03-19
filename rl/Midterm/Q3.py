import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from rl.markov_decision_process import FiniteMarkovDecisionProcess
from typing import (Tuple, Mapping)
from rl.distribution import Categorical
import numpy as np
import matplotlib.pyplot as plt
from rl.dynamic_programming import policy_iteration, value_iteration, greedy_policy_from_vf
from rl.markov_decision_process import StateActionMapping
import math
from dataclasses import dataclass

"""
d: Dict[CareerState, Dict[DaySplit, Categorical[Tuple[CareerState, int]]]] = {}
for w in range(1, self.W+1) :
    for x in range(0, self.W - w + 1) :
        for O in [0, 1] :

            state = CareerState(w, x, O)
            actionMap : Dict[DaySplit, Categorical[Tuple[CareerState, int]]] = {}

            for l in range(self.H + 1) :
                for s in range(self.H+1-l) :
                    action = DaySplit(l, s,self. H-l-s)

                    w2 = min(w + max(state.O, state.x), self.W)
                    distMap = {}

                    poissonSum = 0
                    for y in range(self.W - w2) :
                        for O2 in [0, 1] :
                            state2 = CareerState(w2, y, O2)

                            p = poisson(self.alpha*action.l, y)*((self.beta*action.s/self.H)**O2)*((1-self.beta*action.s/self.H)**(1-O2))
                            poissonSum += p
                            distMap[(state2, action.w*state2.w)] = p

                    y = self.W - w2
                    for O2 in [0, 1] :
                        state2 = CareerState(w2, y, O2)
                        poissonProb = 1-poissonSum
                        p = poissonProb*((self.beta*action.s/self.H)**O2)*((1-self.beta*action.s/self.H)**(1-O2))

                        distMap[(state2, action.w*state2.w)] = p

                    actionMap[action] = Categorical(distMap)
            d[state] = actionMap
"""

StateActionMapping

@dataclass(frozen=True)
class CareerState :
    w : int
    x : int
    O : int

@dataclass(frozen=True)
class DaySplit :
    l : int
    s : int
    w : int

def poisson(lamb, k) :
    return(math.exp(-lamb)*((lamb)**k)/(math.factorial(k)))


class CareerMDP(FiniteMarkovDecisionProcess[int, DaySplit]) :
    def __init__(self, H: int, W: int, alpha: float, beta: float):
        self.H = H
        self.W = W
        self.alpha = alpha
        self.beta = beta
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> StateActionMapping[int, DaySplit]:

        d: Dict[int, Dict[DaySplit, Categorical[Tuple[int, int]]]] = {}

        for w in range(1, self.W+1) :
            actionMap : Dict[DaySplit, Categorical[Tuple[int, int]]] = {}
            for l in range(self.H + 1) :
                for s in range(self.H+1-l) :
                    action = DaySplit(l, s,self. H-l-s)
                    probDict = {}
                    probDict[(w, action.w*w)] = poisson(self.alpha*action.l, 0)*(1-self.beta*action.s/self.H)
                    probDict[(min(self.W, w+1), action.w*w)] = probDict.get((min(self.W, w+1), action.w*w), 0) + poisson(self.alpha*action.l, 1) + poisson(self.alpha*action.l, 0)*(self.beta*action.s/self.H)
                    for k in range(2, self.W - w) :
                        probDict[(w+k, action.w*w)] = poisson(self.alpha*action.l, k)
                    probDict[(self.W, action.w*w)] = probDict.get((self.W, action.w*w), 0) + 1 - sum(probDict.values())
                    actionMap[action] = Categorical(probDict)
            d[w] = actionMap
        return(d)

def value_iteration_optimality(mdp, gamma):
    it = value_iteration(mdp, gamma)
    vf1 = next(it)
    vf2 = next(it)
    while max(abs(np.array(list(vf2.values())) - np.array(list(vf1.values())))) != 0.0 :
        vf3 = next(it)
        vf1 = vf2
        vf2 = vf3
    opt_vf = vf2
    opt_policy = greedy_policy_from_vf(mdp, opt_vf, gamma)
    #print("opt_vf : ", opt_vf)
    #print("opt_policy : ", opt_policy)
    return(opt_vf, opt_policy)

if __name__ == '__main__':
    H, W, alpha, beta = 10, 30, 0.08, 0.82
    mdp = CareerMDP(H, W, alpha, beta)
    print(mdp.get_action_transition_reward_map()[20][DaySplit(l=0, s=0, w=10)])
    print(mdp.get_action_transition_reward_map()[30][DaySplit(l=0, s=0, w=10)])
    opt_vf, opt_policy = value_iteration_optimality(mdp, 0.95)
    states = []
    L, S, W = [], [], []
    for state in opt_policy.states() :
        action = opt_policy.act(state).sample()
        states.append(state)
        L.append(action.l)
        S.append(action.s)
        W.append(H-action.l-action.s)

    plt.plot(states, L, label = "learning")
    plt.plot(states, S, label = "job research")
    plt.plot(states, W, label = "work")
    plt.legend()
    plt.xlabel("Hourly-Wage")
    plt.savefig("Q3.png")
    plt.show()
