from bee_colonies.models.agent import Agent
from gym.spaces import Discrete

Coord = tuple[int, int]

# 0: stay still, 1: move up, 2: move down, 3: move left, 4: move right, 5: attack, 6: pick, 7: drop
BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, BEE_PICK, BEE_DROP, BEE_N_ACTIONS = range(9)

# FINE TUNE ARGUMENTS
BEE_ATTACK_POWER = 2


class Bee(Agent):
    def __init__(self, beehive_id, local_beehive_id):
        super().__init__()
        self.beehive_location = None
        self.is_alive = True
        self.pollen = False  # Indicates if the bee is carrying pollen.
        self.beehive_id = beehive_id
        self.local_beehive_id = local_beehive_id
        self.attack_power = BEE_ATTACK_POWER
        self.action_space = Discrete(BEE_N_ACTIONS)

    def set_spawn(self, spawn_location: Coord):
        self.beehive_location = spawn_location
        return super().set_spawn(spawn_location)

    def action(self) -> int:
        """
        This method should be implemented by the child class.
        For testing purposes, it returns a random action.
        """
        return self.action_space.sample(mask=self.mask)

    def collect_pollen(self):
        """Bee collects pollen from a flower. Since flowers have infinite pollen, just toggle state."""
        if not self.pollen:
            self.pollen = True

    def drop_pollen(self) -> bool:
        """Bee drops off pollen at the beehive."""
        if self.pollen:
            self.pollen = False
            return True
        return False

    def __repr__(self):
        rep = f"B{self.beehive_id}.{self.local_beehive_id}"
        if not self.is_alive:
            rep = f"[{rep}]"
        else:
            rep += "<>" if self.pollen else ""
        return rep
