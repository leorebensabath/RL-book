import sys
sys.path.append("/Users/leore/Library/Mobile Documents/com~apple~CloudDocs/Desktop/StanfordCourses/CME241/RL-book")

from rl.chapter9.order_book import OrderBook
from rl.chapter9.order_book import DollarsAndShares, PriceSizePairs
from rl.markov_process import MarkovProcess
from typing import Sequence, Tuple, Optional, List
from rl.distribution import Categorical
import numpy as np
import math

class stateOrderBook :
    def __init__(self,orderBook) :
        self.ob = orderBook

class OrderBookDynamics(MarkovProcess[OrderBook]) :

    def __init__(self, max_price, min_price) :
        self.max_price = max_price
        self.min_price = min_price

    def transition(self, state) :
        bid = state.ob.bid_price()
        ask = state.ob.ask_price()

        #sell
        distribution_sell = {}
        for p in range(ask, self.max_price + 1) :
            _, new_book = state.ob.sell_limit_order(p, 10)
            distribution_sell[stateOrderBook(new_book)] = (ask**p)*np.exp(-ask)/math.factorial(p)
        #buy
        distribution_buy = {}
        for p in range(self.min_price, bid+1) :
            _, new_book = state.ob.buy_limit_order(p, 10)
            distribution_buy[stateOrderBook(new_book)] = (bid**p)*np.exp(-bid)/math.factorial(p)

        #sell_MO
        _, new_book = state.ob.sell_market_order(10)
        distribution_sell[stateOrderBook(new_book)] = 0.4

        #buy_MO
        _, new_book = state.ob.buy_market_order(10)
        distribution_buy[stateOrderBook(new_book)] = 0.4

        return(Categorical(distribution_sell.update(distribution_buy)))
