from bee_colonies.env.bee_colonies import WASP_DOWN, WASP_UP, WASP_RIGHT, WASP_LEFT, WASP_STAY
from bee_colonies.models.wasp import Wasp


class GreedyWasp(Wasp):
    def __init__(self, id, attack_power=1):
        super().__init__(id, attack_power)
        self.target_beehive = None  # Initially, no target is selected.

    def select_target(self, beehive_coordinates):
        # This method selects the nearest beehive as the target
        if not beehive_coordinates:
            return None
        self.target_beehive = min(beehive_coordinates, key=lambda x: self.__distance(self.position, x))
        return self.target_beehive

    def action(self, beehive_coordinates):
        # Determines the next action towards the selected target
        if not self.target_beehive or self.target_beehive not in beehive_coordinates:
            self.select_target(beehive_coordinates)

        # Example decision process to move towards the target
        target_x, target_y = self.target_beehive
        x, y = self.position
        if x < target_x:
            return WASP_DOWN
        elif x > target_x:
            return WASP_UP
        elif y < target_y:
            return WASP_RIGHT
        elif y > target_y:
            return WASP_LEFT
        return WASP_STAY  # Default to staying still if already at the target

    def __distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
