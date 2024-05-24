import numpy as np
from bee_colonies.models.queen_bee import QueenBee
from bee_colonies.models.agent import Agent
from gym.spaces import Discrete
from config import get_config

CONFIG = get_config()
# 0: stay still, 1: move up, 2: move down, 3: move left, 4: move right, 5: attack
WASP_STAY, WASP_UP, WASP_DOWN, WASP_LEFT, WASP_RIGHT, WASP_ATTACK, WASP_N_ACTIONS = range(7)

# FINE TUNE ARGUMENTS
WASP_LIFE_POINTS = CONFIG["wasp_life_points"]
WASP_ATTACK_POWER = CONFIG["wasp_attack_power"]


Coord = tuple[int, int]

class Wasp(Agent):
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.health = WASP_LIFE_POINTS
        self.is_alive = True
        self.attack_power = WASP_ATTACK_POWER
        self.action_space = Discrete(WASP_N_ACTIONS)

    def receive_damage(self, damage):
        """
        Method to apply damage to the wasp. It reduces the health by the damage amount.
        If the health drops to 0 or below, it marks the wasp as not alive.
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0  # Ensure health doesn't go negative.
            self.is_alive = False

    def attack_beehive(self, queen_bee: QueenBee):
        """Attacks the specified beehive."""
        if queen_bee:
            queen_bee.receive_damage(self.attack_power)

    def action(self) -> int:
        """
        This method should be implemented by the child class.
        For testing purposes, it returns a random action.
        """
        return self.action_space.sample(mask=self.mask)

    def __repr__(self):
        """
        Represent the Wasp object with its ID. If the wasp is not alive, enclose its ID in brackets.
        """
        rep = f"W{self.id}"
        if not self.is_alive:
            rep = f"[{rep}]"
        return rep

def move_towards(src: Coord, dest: Coord):
    x1, y1 = src
    x2, y2 = dest
    dx, dy = abs(x2 - x1), abs(y2 - y1)

    if dx == 0 and dy == 0:
        return WASP_STAY
    if dx > dy:
        return WASP_UP if x2 < x1 else WASP_DOWN
    else:
        return WASP_LEFT if y2 < y1 else WASP_RIGHT
