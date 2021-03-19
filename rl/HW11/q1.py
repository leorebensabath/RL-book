from typing import Callable,  Iterable, Iterator, Mapping, TypeVar
import itertools
import rl.markov_process as mp
S = TypeVar('S')


def tabular_mc(traces: Iterable[Iterable[mp.TransitionStep[S]]], f: Callable, approx0 : list)  -> Iterator[Mapping]:
    def mc_update(func_approx, step) :
        occurences, vf_estimate = func_approx[0], func_approx[1]
        state, Y = step.state, step.return_
        occurences[state] = occurences.get(state, 0) + 1
        vf_estimate[state] = (1 -f(occurences[state]))*vf_estimate.get(state,0) + f(occurences[state])*Y
        return((occurences, vf_estimates))

    def traces_iterable(traces) :
        for trace in traces :
            for step in trace :
                yield step

    vf_estimates = {}
    occurences = {}

    return itertools.accumulate(traces_iterable(traces), func=mc_update)
