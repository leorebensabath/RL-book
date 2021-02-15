import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from rl.HW3.Q3 import FrogAndLilypadsMDP
from rl.dynamic_programming import policy_iteration

def policy_iteration_optimality(n: int, thresh : float):
    fl_mdp = FrogAndLilypadsMDP(n)
    it = policy_iteration(mdp = fl_mdp, gamma = 1)
    vf1, pol1 = next(it)
    vf2, pol2 = next(it)
    while max(np.array(list(vf2.values())) - np.array(list(vf1.values()))) > thresh :
        vf3, pol3 = next(it)
        vf1, vf2 = vf2, vf3
    return((vf2, pol2))

def value_iteration_optimality(n: int, thresh : float):
    fl_mdp = FrogAndLilypadsMDP(n)
    it = value_iteration(mdp = fl_mdp, gamma = 1)
    vf1 = next(it)
    vf2 = next(it)
    while max(np.array(list(vf2.values())) - np.array(list(vf1.values()))) > thresh :
        vf3, pol3 = next(it)
        vf1, vf2 = vf2, vf3
    return(vf2)

if __name__ == '__main__':
    n=5
    fl_mdp = FrogAndLilypadsMDP(n)
    it = policy_iteration(mdp = fl_mdp, gamma = 1)

    for i in range(10) :
        print(next(it))
