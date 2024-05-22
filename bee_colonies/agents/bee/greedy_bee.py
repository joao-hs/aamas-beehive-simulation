import numpy as np
from bee_colonies.models.bee import Bee, BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, BEE_PICK, \
    BEE_DROP, BEE_N_ACTIONS, move_away, move_towards
from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance



class GreedyBee(Bee):
    def __init__(self, local_beehive_id):
        super().__init__(local_beehive_id)

    def action(self) -> int:
        """
        Much like greedy bees, but coordinate on flowers pursuit
        """
        if not self.is_alive:
            return apply_mask_to_action(BEE_STAY, self.mask)
        
        # Always prioritize attacking
        if self.mask[BEE_ATTACK] == 1:
            return apply_mask_to_action(BEE_ATTACK, self.mask)

        stay = np.zeros(BEE_N_ACTIONS)
        stay[BEE_STAY] = 1
        if np.array_equal(self.mask, stay):
            return BEE_STAY
        position = self.last_observation["position"]

        if self.pollen:
            if position == self.beehive_location:
                return apply_mask_to_action(BEE_DROP, self.mask)
            return apply_mask_to_action(move_towards(position, self.beehive_location), self.mask)
        
        visible_flowers = self.last_observation["flowers"]
        if len(visible_flowers) == 0:
            return apply_mask_to_action(move_away(position, self.beehive_location), self.mask)
        
        visible_flowers.sort(key=lambda x: manhattan_distance(position, x.position))
        
        for f in visible_flowers:
            if f.pollen:
                # if bee on flower
                if position == f.position:
                    return apply_mask_to_action(BEE_PICK, self.mask)
                return apply_mask_to_action(move_towards(position, f.position), self.mask)
            
        return apply_mask_to_action(move_away(position, self.beehive_location), self.mask)

