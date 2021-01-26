import sys
sys.path.append("/Users/leore/Desktop/StanfordCourses/CME241/RL-book")

from rl.markov_process import Transition, FiniteMarkovProcess
from rl.distribution import Categorical
from typing import Mapping, Dict
from dataclasses import dataclass
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import RewardTransition

@dataclass(frozen=True)
class PlayerState :
    position : int

    def player_position(self) -> int :
        return(self.position)

class SnackLaddersMRPFinite(FiniteMarkovRewardProcess[PlayerState]) :

        def __init__(self, snakes_ladders_map, game_size) :
            self.snakes_ladders_map = snakes_ladders_map
            self.game_size = game_size
            super().__init__(self.get_transition_reward_map())

        def get_transition_reward_map(self) -> RewardTransition[PlayerState]:
            d: Dict[PlayerState, Categorical[Tuple[PlayerState, float]]] = {}
            for position in range(1, game_size) :
                start_state = PlayerState(position)
                state_probs_map: Mapping[PlayerState, float] = {}
                for destination in range(position + 1, min(position+7, self.game_size+1)) :
                    end_state = PlayerState(self.snakes_ladders_map.get(destination, destination))
                    state_probs_map[(end_state, 1)] = 1/6 + (1/6)*(max(0, 6-(100-position)) if destination == 100 else 0)
                d[start_state] = Categorical(state_probs_map)
            d[PlayerState(game_size)] = None
            return d

if __name__ == '__main__':
    game_size = 100
    snakes_ladders_map = {3:39, 7:48, 12:51, 20:41, 25:57, 28:35, 31:6, 38:1, 45:74, 49:8, 53:17, 60:85, 67:90, 69:92, 70:34, 76:37, 77:83, 82:63, 88:50, 94:42, 98:54}
    sl_mrp = SnackLaddersMRPFinite(snakes_ladders_map, game_size)

    v = sl_mrp.get_value_function_vec(1)
    print(v[0])
