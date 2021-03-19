import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from rl.markov_decision_process import FiniteMarkovDecisionProcess
from typing import (Tuple, Mapping)
from rl.distribution import Categorical
import numpy as np
import matplotlib.pyplot as plt
from rl.dynamic_programming import policy_iteration, value_iteration, greedy_policy_from_vf
from rl.markov_decision_process import StateActionMapping

UP, DOWN, LEFT, RIGHT = 'UP', 'DOWN', 'LEFT', 'RIGHT'
actions = {UP : (-1,0), DOWN : (1,0), LEFT : (0,-1), RIGHT:(0,1)}
SPACE, BLOCK, GOAL = 'SPACE', 'BLOCK', 'GOAL'

maze_grid = {(0, 0): SPACE, (0, 1): BLOCK, (0, 2): SPACE, (0, 3): SPACE, (0, 4): SPACE,
             (0, 5): SPACE, (0, 6): SPACE, (0, 7): SPACE, (1, 0): SPACE, (1, 1): BLOCK,
             (1, 2): BLOCK, (1, 3): SPACE, (1, 4): BLOCK, (1, 5): BLOCK, (1, 6): BLOCK,
             (1, 7): BLOCK, (2, 0): SPACE, (2, 1): BLOCK, (2, 2): SPACE, (2, 3): SPACE,
             (2, 4): SPACE, (2, 5): SPACE, (2, 6): BLOCK, (2, 7): SPACE, (3, 0): SPACE,
             (3, 1): SPACE, (3, 2): SPACE, (3, 3): BLOCK, (3, 4): BLOCK, (3, 5): SPACE,
             (3, 6): BLOCK, (3, 7): SPACE, (4, 0): SPACE, (4, 1): BLOCK, (4, 2): SPACE,
             (4, 3): BLOCK, (4, 4): SPACE, (4, 5): SPACE, (4, 6): SPACE, (4, 7): SPACE,
             (5, 0): BLOCK, (5, 1): BLOCK, (5, 2): SPACE, (5, 3): BLOCK, (5, 4): SPACE,
             (5, 5): BLOCK, (5, 6): SPACE, (5, 7): BLOCK, (6, 0): SPACE, (6, 1): BLOCK,
             (6, 2): BLOCK, (6, 3): BLOCK, (6, 4): SPACE, (6, 5): BLOCK, (6, 6): SPACE,
             (6, 7): SPACE, (7, 0): SPACE, (7, 1): SPACE, (7, 2): SPACE, (7, 3): SPACE,
             (7, 4): SPACE, (7, 5): BLOCK, (7, 6): BLOCK, (7, 7): GOAL}



class MazeMDP1(FiniteMarkovDecisionProcess[Tuple[int, int], str]) :
    def __init__(self, maze_grid : Mapping[Tuple[int, int], str]):
        self.maze_grid = maze_grid
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> StateActionMapping[Tuple[int, int], str]:

        d: Dict[Tuple[int, int], Dict[str, Categorical[Tuple[Tuple[int, int], int]]]] = {}

        for square, type in self.maze_grid.items() :
            if type == GOAL :
                d[square] = None
            elif type == SPACE :
                actionMap : Dict[str, Categorical[Tuple[Tuple[int, int], int]]] = {}
                for action, move in actions.items() :
                    squareNext = (square[0]+move[0], square[1]+move[1])
                    typeNext = self.maze_grid.get(squareNext, BLOCK)
                    if ((typeNext == SPACE) or (typeNext == GOAL)) :
                        actionMap[action] = Categorical({(squareNext, -1):1})
                d[square] = actionMap
        return(d)

class MazeMDP2(FiniteMarkovDecisionProcess[Tuple[int, int], str]) :
    def __init__(self, maze_grid : Mapping[Tuple[int, int], str]):
        self.maze_grid = maze_grid
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> StateActionMapping[Tuple[int, int], str]:

        d: Dict[Tuple[int, int], Dict[str, Categorical[Tuple[Tuple[int, int], int]]]] = {}

        for square, type in self.maze_grid.items() :
            if type == GOAL :
                d[square] = None
            elif type == SPACE :
                actionMap : Dict[str, Categorical[Tuple[Tuple[int, int], int]]] = {}
                for action, move in actions.items() :
                    squareNext = (square[0]+move[0], square[1]+move[1])
                    typeNext = self.maze_grid.get(squareNext, BLOCK)
                    if (typeNext == SPACE) :
                        actionMap[action] = Categorical({(squareNext, 0):1})
                    elif (typeNext == GOAL) :
                        actionMap[action] = Categorical({(squareNext, 1):1})
                d[square] = actionMap
        return(d)

def value_iteration_optimality(mdp, gamma):
    it = value_iteration(mdp, gamma)
    vf1 = next(it)
    vf2 = next(it)
    counter = 2
    while max(abs(np.array(list(vf2.values())) - np.array(list(vf1.values())))) != 0.0 :
        vf3 = next(it)
        vf1 = vf2
        vf2 = vf3
        counter+=1
    opt_vf = vf2
    opt_policy = greedy_policy_from_vf(mdp, opt_vf, 1)
    print("opt_vf : ", opt_vf)
    print("opt_policy : ", opt_policy)
    print("counter : ", counter)
    return(opt_vf, opt_policy, counter)

if __name__ == '__main__':

    mdp1 = MazeMDP1(maze_grid)
    opt_vf1, opt_policy1, counter1 = value_iteration_optimality(mdp1, 1)

    mdp2 = MazeMDP2(maze_grid)
    opt_vf2, opt_policy2, counter2 = value_iteration_optimality(mdp2, 0.5)
