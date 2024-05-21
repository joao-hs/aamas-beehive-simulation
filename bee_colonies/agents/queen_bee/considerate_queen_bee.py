import math
from copy import copy

import numpy as np

from bee_colonies.models.agent import apply_mask_to_action
from bee_colonies.models.queen_bee import QueenBee, HEALTH_SCORE_FUNCTION, IS_GOOD_HEALTH, IS_BAD_HEALTH
from bee_colonies.models.bee import Bee

GOOD_HEALTH_BEE_PERCENT = 10
OK_HEALTH_BEE_PERCENT = 25
BAD_HEALTH_BEE_PERCENT = 50


class ConsiderateQueenBee(QueenBee):
    def __init__(self, id: int, bees: list[Bee]):
        super().__init__(id, bees)

    def action(self) -> np.ndarray:
        """
        If wasp is nearby, keep every bee
        If health is good, release GOOD_HEALTH_BEE_PERCENT% of bees
        If health is OK, release OK_HEALTH_BEE_PERCENT% of bees
        If health is bad, release BAD_HEALTH_BEE_PERCENT% of bees
        """
        if not self.is_alive:
            return apply_mask_to_action(np.zeros(self.action_space.n, dtype=np.int8), self.mask)
        if len(self.last_observation["wasps"]) != 0:
            return apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8), self.mask)
        health_score = HEALTH_SCORE_FUNCTION(self.food_quantity, self.alive_bees)
        if IS_GOOD_HEALTH(health_score):
            return apply_mask_to_action(self.__release_x_percent(GOOD_HEALTH_BEE_PERCENT), self.mask)
        elif IS_BAD_HEALTH(health_score):
            return apply_mask_to_action(self.__release_x_percent(BAD_HEALTH_BEE_PERCENT), self.mask)
        else:
            return apply_mask_to_action(self.__release_x_percent(OK_HEALTH_BEE_PERCENT), self.mask)

    def __release_x_percent(self, x: int):
        action = copy(self.presence_array)
        no_inside_bees = (action == 1).sum()
        count = min(math.ceil(no_inside_bees * x / 10), no_inside_bees)
        action[np.random.choice(np.where(action == 1)[0], count, replace=False)] = 0
        return action
