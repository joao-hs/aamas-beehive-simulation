import math
from copy import copy

import numpy as np

from bee_colonies.models.agent import apply_mask_to_action
from bee_colonies.models.queen_bee import QueenBee, HEALTH_SCORE_FUNCTION, IS_GOOD_HEALTH, IS_BAD_HEALTH
from bee_colonies.models.bee import Bee
from config import get_config

CONFIG = get_config()
KEEP_RATIO_GOOD_HEALTH = CONFIG["keep_ratio_good_health"]
KEEP_RATIO_OK_HEALTH = CONFIG["keep_ratio_ok_health"]
KEEP_RATIO_BAD_HEALTH = CONFIG["keep_ratio_bad_health"]


class ConsiderateQueenBee(QueenBee):
    def __init__(self, id: int, bees: list[Bee], new_bee_class):
        super().__init__(id, bees, new_bee_class)

    def action(self) -> np.ndarray:
        """
        If wasp is nearby, keep every bee
        If health is good, release GOOD_HEALTH_BEE_PERCENT% of bees
        If health is OK, release OK_HEALTH_BEE_PERCENT% of bees
        If health is bad, release BAD_HEALTH_BEE_PERCENT% of bees
        """
        if not self.is_alive:
            return apply_mask_to_action(np.zeros(self.action_space.n, dtype=np.int8), self.mask)
        # in observation, wasp is represented by a tuple (distance, alive)
        nearby_wasps = list(filter(lambda wasp: wasp[1], self.last_observation["wasps"]))
        if len(nearby_wasps) != 0:
            return apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8), self.mask)
        health_score = HEALTH_SCORE_FUNCTION(self.food_quantity, self.alive_bees)
        if IS_GOOD_HEALTH(health_score):
            return apply_mask_to_action(self.__keep_at_least(KEEP_RATIO_GOOD_HEALTH), self.mask)
        elif IS_BAD_HEALTH(health_score):
            return apply_mask_to_action(self.__keep_at_least(KEEP_RATIO_BAD_HEALTH), self.mask)
        else:
            return apply_mask_to_action(self.__keep_at_least(KEEP_RATIO_OK_HEALTH), self.mask)

    def __keep_at_least(self, x: int):
        action = copy(self.presence_array)
        no_inside_bees = (action == 1).sum()
        inside_ratio = no_inside_bees / self.alive_bees
        diff = inside_ratio - x / 100
        if diff > 0:
            # release
            count = min(math.floor(diff * self.alive_bees), no_inside_bees)
            action[np.random.choice(np.where(action == 1)[0], count, replace=False)] = 0
            return action
        # keep
        return np.ones(self.presence_array.size)
