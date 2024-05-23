import numpy as np

from bee_colonies.models.agent import apply_mask_to_action
from bee_colonies.models.bee import Bee
from bee_colonies.models.queen_bee import QueenBee, HEALTH_SCORE_FUNCTION, IS_GOOD_HEALTH


class ConservativeQueenBee(QueenBee):
    def __init__(self, id: int, bees: list[Bee], new_bee_class):
        super().__init__(id, bees, new_bee_class)

    def action(self) -> np.ndarray:
        """
        If there are wasps nearby, keep every bee.
        If the health is good, keep every bee.
        Otherwise, release one bee at a time
        """
        if not self.is_alive:
            return apply_mask_to_action(np.zeros(self.action_space.n, dtype=np.int8), self.mask)
        # in observation, wasp is represented by a tuple (distance, alive)
        nearby_wasps = list(filter(lambda wasp: wasp[1], self.last_observation["wasps"]))
        if len(nearby_wasps) != 0:
            return apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8), self.mask)
        health_score = HEALTH_SCORE_FUNCTION(self.food_quantity, self.alive_bees)
        if IS_GOOD_HEALTH(health_score):
            return apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8), self.mask)
        # choose one present bee
        picked_index = self.presence_array.argmax()
        action = np.ones(self.action_space.n, dtype=np.int8)
        action[picked_index] = 0
        return apply_mask_to_action(action, self.mask)
