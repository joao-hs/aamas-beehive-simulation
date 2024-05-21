from bee_colonies.models.bee import BEE_STAY, Bee, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT
import numpy as np


class RespectfulBee(Bee):
    def __init__(self, id, queen_id, local_beehive_id):
        super().__init__(id, queen_id, local_beehive_id)

    def action(self, observation) -> int:
        # Find the closest flower that this bee can claim
        flower_position, can_claim = self._find_flower_to_claim(observation)

        if can_claim and flower_position:
            # If a flower is claimable, determine the move to get there
            return self._move_towards(flower_position)
        else:
            # Continue searching randomly or perform other behaviors
            return np.random.choice([BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT])
        

    def _find_flower_to_claim(self, observation):
        my_position = self.spawn_location
        closest_flower = None
        min_distance = float('inf')
        can_claim = True

        for flower in observation['flowers']:
            flower_pos = flower  # Assuming each flower is a coordinate
            distance = self.__distance(my_position, flower_pos)
            if distance < min_distance:
                closest_flower = flower
                min_distance = distance
                can_claim = True

                for other_bee in observation['bees']:
                    other_distance = self.__distance(other_bee['position'], flower_pos)
                    if other_distance < distance or (other_distance == distance and other_bee['id'] > self.id):
                        can_claim = False
                        break
            elif distance == min_distance:
                for other_bee in observation['bees']:
                    if other_bee['id'] > self.id:
                        can_claim = False
                        break

        return closest_flower, can_claim


    def _move_towards(self, target_position):
        # Calculate direction to move towards the target
        x_diff = target_position[0] - self.spawn_location[0]
        y_diff = target_position[1] - self.spawn_location[1]
        
        if abs(x_diff) > abs(y_diff):
            return BEE_DOWN if x_diff > 0 else BEE_UP
        else:
            return BEE_RIGHT if y_diff > 0 else BEE_LEFT

    def __distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)