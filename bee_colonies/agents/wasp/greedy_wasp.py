from bee_colonies.env.bee_colonies import WASP_DOWN, WASP_UP, WASP_RIGHT, WASP_LEFT, WASP_STAY
from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance
from bee_colonies.models.wasp import WASP_ATTACK, Wasp, move_towards


class GreedyWasp(Wasp):
    def __init__(self, id, n_clusters, cluster_center_distance):
        super().__init__(id, n_clusters, cluster_center_distance)

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
            return apply_mask_to_action(self.searching_guide.walk_to_cluster(self.last_observation["position"]), self.mask)
    
    def _find_nearest_beehive(self):
        # Filter the list to include only alive beehives before sorting
        alive_beehives = [
            beehive for beehive, is_alive in self.last_observation["beehives"] if is_alive
        ]
        
        if not alive_beehives:
            return None  # Return None if there are no alive beehives
        
        # Now sort the alive beehives by their distance from the wasp's current position
        alive_beehives.sort(key=lambda x: manhattan_distance(self.last_observation["position"], x))
        return alive_beehives[0]  # Return the nearest alive beehive
