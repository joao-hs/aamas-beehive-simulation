from bee_colonies.models.bee import Bee, BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, BEE_PICK, \
    BEE_DROP, move_towards, move_away, random_walk, Coord
from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance

# FINE TUNE ARGUMENTS
KEEP_AWAY_FROM_BEEHIVE_DISTANCE = 5


class SocialBee(Bee):
    def __init__(self, local_beehive_id):
        super().__init__(local_beehive_id)
        self.picked_pollen_from = None
        self.target_flower = None

    def action(self) -> int:
        """
        Much like greedy bees, but coordinate on flowers pursuit
        """
        if not self.is_alive:
            return apply_mask_to_action(BEE_STAY, self.mask)
        position = self.last_observation["position"]

        if self.pollen:
            if position == self.beehive_location:
                self.queen.pursuing_flower_set.remove(self.picked_pollen_from)
                self.target_flower = None
                self.picked_pollen_from = None
                return apply_mask_to_action(BEE_DROP, self.mask)
            return apply_mask_to_action(move_towards(position, self.beehive_location), self.mask)

        if self.target_flower is not None:
            if position == self.target_flower.position:
                self.picked_pollen_from = self.target_flower
                return apply_mask_to_action(BEE_PICK, self.mask)
            return apply_mask_to_action(move_towards(position, self.target_flower.position), self.mask)

        visible_flowers = self.last_observation["flowers"]
        if len(visible_flowers) == 0:
            return apply_mask_to_action(self.search_for_flowers(position), self.mask)

        for flower_coord in visible_flowers:
            if flower_coord not in self.queen.pursuing_flower_set:
                self.target_flower = flower_coord
                return apply_mask_to_action(move_towards(position, self.target_flower.position), self.mask)
        return apply_mask_to_action(self.search_for_flowers(position), self.mask)

    def search_for_flowers(self, position: Coord):
        """
        Random walk but keep distance from beehive
        """
        if manhattan_distance(position, self.beehive_location) < KEEP_AWAY_FROM_BEEHIVE_DISTANCE:
            return move_away(position, self.beehive_location)
        return random_walk()
