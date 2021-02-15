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

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'
actions = {UP : (0,1), DOWN : (0,-1), LEFT : (1,0), RIGHT:(0,1)}


class FrogAndLilypadsMDP(FiniteMarkovDecisionProcess[int, str]) :
    def __init__(self, maze_grid):
        self.maze_grid = maze_grid
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> frogJumpsMapping:

        d: Dict[Tuple[int, int], Dict[str, Categorical[Tuple[Tuple[int, int], int]]]] = {}

        for square, type in self.maze_grid.items() :
            if type == GOAL :
                d[square] = None
            elif type == SPACE :
                actionMap : Dict[str, Categorical[Tuple[Tuple[int, int], int]]] = {}
                for action, move in actions.items() :
                    squareNext = (square[0]+move[0], square[1], move[1])
                    typeNext = self.maze_grid.get(squareNext, BLOCK)
                    if typeNext == SPACE :
                        actionMap[action] = Categorical({(squareNext, 1):1})
                d[square] = actionMap
        return(d)



if __name__ == '__main__':
