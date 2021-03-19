import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from dataclasses import dataclass
from typing import Optional, Mapping, Callable
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovProcess
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes
from rl.chapter2.stock_price_mp import StockPriceMP1

@dataclass(frozen=True)
class StateMP1:
    price: int

@dataclass
class StockPriceMRP1(StockPriceMP1,MarkovRewardProcess[StateMP1]):
    f : Callable

    def __init__(self, f) :
        self.f = f
        super().__init__()

    def transition_reward(
        self,
        state: StateMP1
    ) -> Categorical[Tuple[StateMP1, float]]:
        up_p = self.up_prob(state)

        return Categorical({
            (StateMP1(state.price + 1), self.f(state)): up_p,
            (StateMP1(state.price - 1), self.f(state)): 1 - up_p
        })

    def get_value_function(state: StateMP1) -> float:
        if 
