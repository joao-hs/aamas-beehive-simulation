from bee_colonies.env.bee_colonies import WASP_DOWN, WASP_UP, WASP_RIGHT, WASP_LEFT, WASP_STAY
from bee_colonies.models.wasp import Wasp
import numpy as np

class GreedyWasp(Wasp):
    def __init__(self, id, vision_range=5, initial_health=20):
        super().__init__(id, initial_health=initial_health)
        self.vision_range = vision_range  # Defines how far the wasp can see a beehive
    
    def action(self) -> int:
        # If the wasp can see a beehive, it will choose an action to move towards or attack the beehive
        beehive_location = self._find_nearest_beehive()
        if beehive_location:
            return self._move_towards_beehive(beehive_location)
        else:
            # Move randomly if no beehive is visible
            return np.random.choice([WASP_STAY, WASP_UP, WASP_DOWN, WASP_LEFT, WASP_RIGHT])
    
    def _find_nearest_beehive(self):
        # This method would check the observation to find the nearest beehive within the vision range
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                check_pos = (self.spawn_location[0] + dx, self.spawn_location[1] + dy)
                if check_pos in self.observation['beehives']:  # Assuming observation includes beehive positions
                    return check_pos
        return None

    def _move_towards_beehive(self, beehive_location):
        # Decide the best action to move towards the beehive
        x_diff = beehive_location[0] - self.spawn_location[0]
        y_diff = beehive_location[1] - self.spawn_location[1]
        
        if abs(x_diff) > abs(y_diff):
            return WASP_DOWN if x_diff > 0 else WASP_UP
        else:
            return WASP_RIGHT if y_diff > 0 else WASP_LEFT
