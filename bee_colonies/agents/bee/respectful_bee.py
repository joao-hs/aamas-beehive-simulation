from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance
from bee_colonies.models.bee import BEE_ATTACK, BEE_N_ACTIONS, BEE_STAY, Bee, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, \
    BEE_DROP, BEE_PICK, move_towards
import numpy as np

from bee_colonies.models.searching_guide import SearchingGuide
from config import get_config

CONFIG = get_config()
RANDOM_WALK_INTENT = CONFIG["random_walk_intent"]


class RespectfulBee(Bee):
    def __init__(self, local_beehive_id):
        super().__init__(local_beehive_id)
        self.searching_guide = SearchingGuide([BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT], RANDOM_WALK_INTENT)

    def action(self) -> int:
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

        # Find the closest flower that this bee can claim
        flower, can_claim = self._find_flower_to_claim(self.last_observation)

        if can_claim and flower:
            if position == flower.position:
                return apply_mask_to_action(BEE_PICK, self.mask)
            # If a flower is claimable, determine the move to get there
            return apply_mask_to_action(move_towards(self.last_observation["position"], flower.position), self.mask)
        else:
            # Continue searching randomly or perform other behaviors
            return apply_mask_to_action(self.searching_guide.walk(self.last_observation["position"]), self.mask)

    def _find_flower_to_claim(self, observation):
        my_position = observation['position']
        closest_flower = None
        can_claim = True

        flowers = list(filter(lambda x: x.pollen, observation['flowers']))
        flowers.sort(key=lambda x: manhattan_distance(my_position, x.position))
        claimed = set()

        for flower in flowers:
            flower_pos = flower.position
            distance = manhattan_distance(my_position, flower_pos)
            for other_bee_colony, other_bee_id, other_bee_pos in observation['bees']:
                if other_bee_colony != self.queen_id:
                    continue
                other_distance = manhattan_distance(other_bee_pos, flower_pos)
                if other_distance < distance or (other_distance == distance and other_bee_id not in claimed and other_bee_id < self.local_beehive_id):
                    can_claim = False
                    claimed.add(other_bee_id)
                    break
            if can_claim:
                closest_flower = flower
                break

        return closest_flower, can_claim
