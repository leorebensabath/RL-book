import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from numpy import linalg
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar)
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from itertools import combinations_with_replacement, product
from rl.distribution import Categorical
import numpy as np
import matplotlib.pyplot as plt

frogJumpsMapping = StateActionMapping[int, str]

class FrogAndLilypadsMDP(FiniteMarkovDecisionProcess[int, str]) :
    def __init__(self, n):
        self.n = n
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> frogJumpsMapping:

        d: Dict[int, Dict[str, Categorical[Tuple[int, int]]]] = {}

        d[0], d[self.n] = None, None

        for i in range(1, self.n):
            actionMap : Dict[str, Categorical[Tuple[int, int]]] = {}

            sr_probs_dict_A : Dict[Tuple[int, int], float] = {(i+1, 1): (self.n-i)/n, (i-1, -1): i/self.n}
            actionMap["A"] = Categorical(sr_probs_dict_A)

            sr_probs_dict_B : Dict[Tuple[int, int], float] = {(j, j-i): 1/self.n for j in range(0,self.n+1) if j != i}
            actionMap["B"] = Categorical(sr_probs_dict_B)

            d[i] = actionMap

        return(d)

def get_optimality(n : int) -> Tuple[FinitePolicy, np.ndarray] :
    fl_mdp = FrogAndLilypadsMDP(n)
    print(fl_mdp.get_action_transition_reward_map())
    deterministic_policies = product("AB", repeat=n-1)
    odp = None
    ovf = None
    for prod in deterministic_policies :
        policy_map = {0: None, n: None}
        for i in range(1, n) :
            policy_map[i] = Categorical({prod[i-1]: 1})
        policy = FinitePolicy(policy_map)
        fl_mrp = fl_mdp.apply_finite_policy(policy)
        value_function = fl_mrp.get_value_function_vec(1)
        if odp == None :
            odp = policy
            odp_keys = prod
            ovf = value_function
        else :
            comparison = [(value_function[i] > ovf[i]) for i in range(n-1)]
            if all(comparison) :
                odp = policy
                odp_keys = prod
                ovf = value_function
    return((odp_keys, ovf))

def get_optimal_escape_probability(deterministic_policy: Sequence, n : int) -> Sequence:
    M = np.zeros((n+1,n+1))

    M[0,0] = 1
    M[n, n] = 1
    for i in range(1,n) :
        if deterministic_policy[i-1] == 'A' :
            M[i,i-1] = i/n
            M[i,i+1] = (n-i)/n
        elif deterministic_policy[i-1] == 'B' :
            for j in range(n+1) :
                if j != i :
                    M[i, j] = 1/n

    Eval, Evec = linalg.eig(M)

    for i in range(len(Eval)) :
        if Eval[i] == 1. :
            if Evec[0, i] == 0.:
                escape_prob = Evec[:,i]
                escape_prob = list(escape_prob/escape_prob[-1])

    return(escape_prob)

if __name__ == '__main__':

    for n in [3, 6, 9] :
        odp, ovf = get_optimality(n)
        print(odp)
        print(get_optimal_escape_probability(odp, n))
        oep = get_optimal_escape_probability(odp, n)
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, n), oep[1:n])
        ax.set_xticks(np.arange(1, n))
        for i in range(n-1):
            ax.annotate(odp[i], (i+1, oep[i+1]+0.002))
        ax.set_xlabel("State")
        ax.set_ylabel("Optimal Escape-Probability")
        ax.set_title(f'n={n}')
        plt.savefig(f'HW3/n={n}')
        plt.show()
