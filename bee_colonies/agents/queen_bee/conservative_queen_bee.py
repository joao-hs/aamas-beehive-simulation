import numpy as np

from bee_colonies.models.bee import Bee
from bee_colonies.models.queen_bee import QueenBee, HEALTH_SCORE_FUNCTION, IS_GOOD_HEALTH


class ConservativeQueenBee(QueenBee):
    def __init__(self, id: int, bees: list[Bee]):
        super().__init__(id, bees)

    def action(self) -> np.ndarray:
        """
        If there are wasps nearby, keep every bee.
        If the health is good, keep every bee.
        Otherwise, release one bee at a time
        """
        if not self.is_alive:
            return self.__apply_mask_to_action(np.zeros(self.action_space.n, dtype=np.int8))
        if len(self.last_observation["wasps"]) != 0:
            return self.__apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8))
        health_score = HEALTH_SCORE_FUNCTION(self.food_quantity, self.alive_bees)
        if IS_GOOD_HEALTH(health_score):
            return self.__apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8))
        # choose one present bee
        picked_index = self.presence_array.argmax()
        action = np.ones(self.action_space.n, dtype=np.int8)
        action[picked_index] = 0
        return self.__apply_mask_to_action(action)

    def __apply_mask_to_action(self, action) -> np.ndarray:
        for m, (index, _) in zip(self.mask, enumerate(action)):
            if m == 0 or m == 1:
                action[index] = m
        return action
