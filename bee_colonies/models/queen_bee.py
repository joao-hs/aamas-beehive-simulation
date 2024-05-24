from bee_colonies.models.bee import Bee, BEE_STAY, BEE_ATTACK
from bee_colonies.models.agent import Agent
import numpy as np
from gym.spaces import MultiBinary
from config import get_config

CONFIG = get_config()
# FINE TUNE ARGUMENTS
STARTING_FOOD_QUANTITY_PER_BEE = CONFIG["starting_food_quantity_per_bee"]
CONSUMED_FOOD_PER_TURN_PER_BEE = CONFIG["consumed_food_per_turn_per_bee"]
FOOD_QUANTITY_PER_POLLEN = CONFIG["food_quantity_per_pollen"]
IS_GOOD_HEALTH = lambda health_score: health_score > CONFIG["good_health_threshold"]
IS_BAD_HEALTH = lambda health_score: health_score < CONFIG["bad_health_threshold"]
TENDENCY_THRESHOLD = CONFIG["tendency_threshold"]
HEALTH_SCORE_FUNCTION = lambda food_quantity, no_bees: food_quantity // no_bees if no_bees > 0 else 0


class QueenBee(Agent):
    def __init__(self, id: int, bees: list[Bee], new_bee_class):
        super().__init__()
        n_bees = len(bees)
        self.id = id
        self.is_alive = True
        self.bees = bees
        self.alive_bees = n_bees
        self.presence_array = np.ones(n_bees)
        self.food_quantity = STARTING_FOOD_QUANTITY_PER_BEE * n_bees
        self.received = 0
        self.last_observation = None
        self.mask = None
        self.action_space = MultiBinary(n_bees)
        self.health_tendency_counter = 0
        self.new_bee = new_bee_class
        # used by social bees
        self.pursuing_flower_map = dict()
        self.section_size = None

    def action(self) -> np.ndarray:
        """
        This method should be implemented by the child class.
        For testing purposes, it returns a random action.
        """
        return self.action_space.sample(mask=self.mask)

    def __repr__(self):
        return f"Queen Bee {self.id} with {self.food_quantity} units of food"

    def receive_damage(self, damage):
        """Reduces the health of the Queen Bee by the specified damage amount."""
        self.food_quantity -= damage
        # if food is less than 0, the beehive dies in the next timestep

    def receive_polen(self):
        """Queen Bee receives polen from a bee."""
        self.received += 1
        self.food_quantity += FOOD_QUANTITY_PER_POLLEN

    def timestep(self) -> tuple[Bee, bool]:
        """Queen Bee's health decreases by the consumed food per turn per bee."""
        if not self.is_alive:
            return None, False
        total_no_bees = len(self.bees)
        if self.alive_bees <= 0:
            self.is_alive = False
            self.__purge_bees()
            return None, False
        self.food_quantity -= CONSUMED_FOOD_PER_TURN_PER_BEE * self.alive_bees
        if self.food_quantity < 0:
            self.is_alive = False
            self.__purge_bees()
            return None, False
        health_score = HEALTH_SCORE_FUNCTION(self.food_quantity, self.alive_bees)
        if IS_GOOD_HEALTH(health_score):
            self.health_tendency_counter = max(self.health_tendency_counter + 1, 1)
        elif IS_BAD_HEALTH(health_score):
            self.health_tendency_counter = min(self.health_tendency_counter - 1, -1)
        else:
            self.health_tendency_counter = 0
        if self.health_tendency_counter >= TENDENCY_THRESHOLD:
            self.presence_array = np.append(self.presence_array, 1)
            new_bee = self.new_bee(total_no_bees)
            new_bee.set_queen(self)
            self.action_space = MultiBinary(total_no_bees + 1)
            self.alive_bees += 1
            return new_bee, True
        elif self.health_tendency_counter <= -TENDENCY_THRESHOLD:
            # sacrifices a bee, preferably if they're inside the beehive
            picked_index = self.__pick_bee_to_sacrifice()
            self.dead_bee(picked_index)
            return self.bees[picked_index], False
        return None, False

    def welcome(self, bee: Bee):
        bee.mask = np.zeros(bee.action_space.n)
        bee.mask[BEE_STAY] = 1
        bee.mask[BEE_ATTACK] = 1
        self.presence_array[bee.local_beehive_id] = 1

    def dead_bee(self, id: int):
        self.alive_bees -= 1
        self.presence_array[id] = 0

    def __purge_bees(self):
        self.alive_bees = 0
        for bee in self.bees:
            bee.is_alive = False

    def __pick_bee_to_sacrifice(self):
        picked_bee_id = None
        for bee in self.bees:
            if bee.is_alive:
                if picked_bee_id is None:
                    picked_bee_id = bee.local_beehive_id  # first alive bee found
                if self.presence_array[bee.local_beehive_id] == 1:
                    return bee.local_beehive_id  # first alive bee inside the beehive
        return picked_bee_id
