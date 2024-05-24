import numpy as np

from bee_colonies.models.agent import apply_mask_to_action, manhattan_distance
import numpy as np
from bee_colonies.models.agent import Agent
from gym.spaces import Discrete
from config import get_config

CONFIG = get_config()

Coord = tuple[int, int]

# 0: stay still, 1: move up, 2: move down, 3: move left, 4: move right, 5: attack, 6: pick, 7: drop
BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, BEE_PICK, BEE_DROP, BEE_N_ACTIONS = range(9)

# FINE TUNE ARGUMENTS
BEE_ATTACK_POWER = CONFIG["bee_attack_power"]


class Bee(Agent):
    def __init__(self, local_beehive_id):
        super().__init__()
        self.beehive_location = None
        self.is_alive = True
        self.pollen = False  # Indicates if the bee is carrying pollen.
        self.queen_id = None
        self.queen = None
        self.local_beehive_id = local_beehive_id
        self.attack_power = BEE_ATTACK_POWER
        self.action_space = Discrete(BEE_N_ACTIONS)

    def set_queen(self, queen):
        self.queen_id = queen.id
        self.queen = queen
        self.set_spawn(queen.spawn_location)
        self.beehive_location = self.spawn_location

    def set_spawn(self, spawn_location: Coord):
        super().set_spawn(spawn_location)
        self.beehive_location = spawn_location

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
        rep = f"B{self.queen_id}.{self.local_beehive_id}"
        if not self.is_alive:
            rep = f"[{rep}]"
        else:
            rep += "<>" if self.pollen else ""
        return rep


def move_towards(src: Coord, dest: Coord):
    x1, y1 = src
    x2, y2 = dest
    dx, dy = abs(x2 - x1), abs(y2 - y1)

    if dx == 0 and dy == 0:
        return BEE_STAY
    if dx > dy:
        return BEE_UP if x2 < x1 else BEE_DOWN
    else:
        return BEE_LEFT if y2 < y1 else BEE_RIGHT

def move_away(src: Coord, away: Coord):
    x1, y1 = src
    x2, y2 = away
    dx, dy = x2 - x1, y2 - y1

    if dx > dy:
        return BEE_UP if dx > 0 else BEE_RIGHT
    else:
        return BEE_LEFT if dy > 0 else BEE_LEFT
