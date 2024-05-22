from bee_colonies.env.bee_colonies import WASP_DOWN, WASP_UP, WASP_RIGHT, WASP_LEFT, WASP_STAY
from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance
from bee_colonies.models.wasp import WASP_ATTACK, Wasp, move_towards
from bee_colonies.models.searching_guide import SearchingGuide
import numpy as np

RANDOM_WALK_INTENT = 3

class GreedyWasp(Wasp):
    def __init__(self, id):
        super().__init__(id)
        self.searching_guide = SearchingGuide([WASP_UP, WASP_DOWN, WASP_LEFT, WASP_RIGHT], RANDOM_WALK_INTENT)
    
    def action(self) -> int:
        # If the wasp can see a beehive, it will choose an action to move towards or attack the beehive
        if not self.is_alive:
            return apply_mask_to_action(WASP_STAY, self.mask)

        beehive_location = self._find_nearest_beehive()
        position = self.last_observation["position"]
        
        if beehive_location:
            if position == beehive_location:
                return apply_mask_to_action(WASP_ATTACK, self.mask)
            return apply_mask_to_action(move_towards(position, beehive_location), self.mask)

        else:
            # Move randomly if no beehive is visible
            return apply_mask_to_action(self.searching_guide.walk(self.last_observation["position"]), self.mask)
    
    def _find_nearest_beehive(self):
        # This method would check the observation to find the nearest beehive within the vision range
        self.last_observation["beehives"].sort(key=lambda x: manhattan_distance(self.last_observation["position"], x))
        return self.last_observation["beehives"][0] if self.last_observation["beehives"] else None