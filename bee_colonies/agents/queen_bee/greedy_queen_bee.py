import numpy as np

from bee_colonies.models.agent import apply_mask_to_action
from bee_colonies.models.queen_bee import QueenBee
from bee_colonies.models.bee import Bee


class GreedyQueenBee(QueenBee):
    def __init__(self, id: int, bees: list[Bee], new_bee_class):
        super().__init__(id, bees, new_bee_class)

    def action(self) -> np.ndarray:
        """
        Unless wasp is nearby, release every bee
        """
        if not self.is_alive:
            return apply_mask_to_action(np.zeros(self.action_space.n, dtype=np.int8), self.mask)
        # in observation, wasp is represented by a tuple (distance, alive)
        nearby_wasps = list(filter(lambda wasp: wasp[1], self.last_observation["wasps"]))
        if len(nearby_wasps) != 0:
            return apply_mask_to_action(np.ones(self.action_space.n, dtype=np.int8), self.mask)
        return apply_mask_to_action(np.zeros(self.action_space.n, dtype=np.int8), self.mask)
