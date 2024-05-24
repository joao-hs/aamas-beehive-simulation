import numpy as np
from bee_colonies.models.bee import BEE_N_ACTIONS, Bee, BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, \
    BEE_PICK, \
    BEE_DROP, move_towards, move_away, Coord
from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance
from bee_colonies.models.searching_guide import SearchingGuide
from config import get_config

# FINE TUNE ARGUMENTS
CONFIG = get_config()
KEEP_AWAY_FROM_BEEHIVE_DISTANCE = CONFIG["keep_away_from_beehive_distance"]
RANDOM_WALK_INTENT = CONFIG["random_walk_intent"]


class SocialBee(Bee):
    def __init__(self, local_beehive_id):
        super().__init__(local_beehive_id)
        self.picked_pollen_from = None
        self.target_flower = None
        self.searching_guide = SearchingGuide([BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT], RANDOM_WALK_INTENT)

    def action(self) -> int:
        """
        Much like greedy bees, but coordinate on flowers pursuit
        """
        if not self.is_alive:
            return apply_mask_to_action(BEE_STAY, self.mask)
        position = self.last_observation["position"]

        # Always prioritize attacking
        if self.mask[BEE_ATTACK] == 1:
            return apply_mask_to_action(BEE_ATTACK, self.mask)

        stay = np.zeros(BEE_N_ACTIONS)
        stay[BEE_STAY] = 1
        if np.array_equal(self.mask, stay):
            return BEE_STAY

        if self.pollen:
            if position == self.beehive_location:
                self.queen.pursuing_flower_map[self.__get_section(self.picked_pollen_from.position)].remove(self.picked_pollen_from)
                self.target_flower = None
                self.picked_pollen_from = None
                return apply_mask_to_action(BEE_DROP, self.mask)
            return apply_mask_to_action(move_towards(position, self.beehive_location), self.mask)

        if self.target_flower is not None:
            if position == self.target_flower.position:
                self.picked_pollen_from = self.target_flower
                return apply_mask_to_action(BEE_PICK, self.mask)
            return apply_mask_to_action(move_towards(position, self.target_flower.position), self.mask)

        visible_flowers = list(filter(lambda x: x.pollen, self.last_observation["flowers"]))
        if len(visible_flowers) == 0:
            return apply_mask_to_action(self.search_for_flowers(position), self.mask)

        visible_flowers.sort(key=lambda x: manhattan_distance(position, x.position))
        for flower in visible_flowers:
            if flower not in self.queen.pursuing_flower_map[self.__get_section(flower.position)]:
                self.target_flower = flower
                self.queen.pursuing_flower_map[self.__get_section(self.target_flower.position)].add(self.target_flower)
                return apply_mask_to_action(move_towards(position, self.target_flower.position), self.mask)
        return apply_mask_to_action(self.search_for_flowers(position), self.mask)

    def __get_section(self, position):
        return (position[0] // self.queen.section_size)*self.queen.section_size, \
            (position[1] // self.queen.section_size)*self.queen.section_size

    def search_for_flowers(self, position: Coord):
        """
        Random walk but keep distance from beehive
        """
        if manhattan_distance(position, self.beehive_location) < KEEP_AWAY_FROM_BEEHIVE_DISTANCE:
            return move_away(position, self.beehive_location)
        return self.searching_guide.walk(position)
