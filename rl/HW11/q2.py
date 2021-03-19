from typing import Callable,  Iterable, Iterator, Mapping, TypeVar
import itertools
import rl.markov_process as mp
S = TypeVar('S')


def tabular_mc(traces: Iterable[Iterable[mp.TransitionStep[S]]], gamma : float, alpha : float)  -> Iterator[Mapping]:
    def td_update(vf_estimates, step) :
        state, R, next_state = step.state, step.reward, step.next_state
        vf_estimate[state] = vf_estimate.get(state,0) + alpha*(R+gamma*vf_estimate.get(next_state,0) - vf_estimate.get(state,0))
        return(vf_estimates)

    def traces_iterable(traces) :
        for trace in traces :
            for step in trace :
                yield step

    vf_estimates = {}
    occurences = {}

    return itertools.accumulate(traces_iterable(traces), func=td_update)
