import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from rl.markov_process import Transition, FiniteMarkovProcess
from rl.distribution import Categorical
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar)
from dataclasses import dataclass
from rl.chapter2.stock_price_simulations import plot_single_trace_all_processes
from rl.gen_utils.plot_funcs import plot_list_of_curves
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class PlayerState :
    position : int

    def player_position(self) -> int :
        return(self.position)

class SnackLaddersMPFinite(FiniteMarkovProcess[PlayerState]) :

    def __init__(self, snakes_ladders_map, game_size) :
        self.snakes_ladders_map = snakes_ladders_map
        self.game_size = game_size
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Transition[PlayerState] :
        d: Dict[PlayerState, Categorical[PlayerState]] = {}
        for position in range(1, game_size) :
            start_state = PlayerState(position)
            state_probs_map: Mapping[PlayerState, float] = {}
            for destination in range(position + 1, min(position+7, self.game_size+1)) :
                end_state = PlayerState(self.snakes_ladders_map.get(destination, destination))
                state_probs_map[end_state] = 1/6 + (1/6)*(max(0, 6-(100-position)) if destination == 100 else 0)
            d[start_state] = Categorical(state_probs_map)
        d[PlayerState(game_size)] = None
        return d

def generate_trace(process_traces : Iterable[Iterable[PlayerState]]) -> Sequence[int] :
    transition_map = sl_mp.get_transition_map()

    trace = next(process_traces)
    state = next(trace)
    trace_points = [state.player_position()]
    while transition_map[state] is not None:
        state = next(trace)
        trace_points.append(state.player_position())
    return(trace_points)


def generate_time_steps_distribution(process_traces : Iterable[Iterable[PlayerState]], n: int) -> Tuple[Sequence[int], Sequence[int]] :
    '''Generate a histogram of the number of time steps required to finish the game based on n sampling of the process
    '''
    distribution_sampling = []
    for i in range(n) :
        trace = generate_trace(process_traces)
        distribution_sampling.append(len(trace)-1)

    pairs = sorted(
        list(Counter(distribution_sampling).items()),
        key=itemgetter(0)
    )
    return([x for x, _ in pairs], [y for _, y in pairs])

if __name__ == '__main__':

    game_size = 100
    snakes_ladders_map = {3:39, 7:48, 12:51, 20:41, 25:57, 28:35, 31:6, 38:1, 45:74, 49:8, 53:17, 60:85, 67:90, 69:92, 70:34, 76:37, 77:83, 82:63, 88:50, 94:42, 98:54}
    sl_mp = SnackLaddersMPFinite(snakes_ladders_map, game_size)

    #print(sl_mp)

    #Plot some traces
    start_d = Categorical({PlayerState(1):1})
    process_traces=sl_mp.traces(start_d)

    list_traces = []

    for i in range(5) :
        trace_points = generate_trace(process_traces)
        list_traces.append(trace_points)
    x_values = []
    plot_list_of_curves([range(1, len(trace)+1) for trace in list_traces], list_traces, ['b', 'g', 'r', 'y', 'black'], \
    [f'Trace {i}' for i in range(len(list_traces))], "Number of time steps", "Square on the board", "Snakes and Ladders game traces")

    #Generate number of time steps distribution :
    hist = generate_time_steps_distribution(process_traces, (10000))
    plt.bar(hist[0], hist[1], width=1)
    plt.xlabel("Number of steps")
    plt.ylabel("Counts")
    plt.title("Number of time steps; Traces = 10000 ")
    plt.show()
