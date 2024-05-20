from bee_colonies.models.bee import Bee
from bee_colonies.models.agent import Agent
import numpy as np
from gym.spaces import MultiBinary

# FINE TUNE ARGUMENTS
STARTING_FOOD_QUANTITY_PER_BEE = 100
CONSUMED_FOOD_PER_TURN_PER_BEE = 1
FOOD_QUANTITY_PER_POLEN = 20
IS_GOOD_HEALTH = lambda health_score: health_score > 25
IS_BAD_HEALTH = lambda health_score: health_score < 15
TENDENCY_THRESHOLD = 10
HEALTH_SCORE_FUNCTION = lambda food_quantity, no_bees: food_quantity // no_bees


class QueenBee(Agent):
    def __init__(self, id: int, bees: list[Bee]):
        super().__init__()
        n_bees = len(bees)
        self.id = id
        self.is_alive = True
        self.bees = bees
        self.presence_array = np.ones(n_bees)
        self.food_quantity = STARTING_FOOD_QUANTITY_PER_BEE * n_bees
        self.last_observation = None
        self.mask = None
        self.action_space = MultiBinary(n_bees)
        self.health_tendency_counter = 0

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

    def receive_polen(self):
        """Queen Bee receives polen from a bee."""
        self.food_quantity += FOOD_QUANTITY_PER_POLEN
        print(f"Queen Bee {self.id} received polen and now has {self.food_quantity} health.")

    def timestep(self) -> tuple[Bee, bool]:
        """Queen Bee's health decreases by the consumed food per turn per bee."""
        if not self.is_alive:
            return None, False
        no_bees = len(self.bees)
        if no_bees == 0:
            self.is_alive = False
            return None, False
        self.food_quantity -= CONSUMED_FOOD_PER_TURN_PER_BEE * no_bees
        health_score = HEALTH_SCORE_FUNCTION(self.food_quantity, no_bees)
        if IS_GOOD_HEALTH(health_score):
            self.health_tendency_counter = max(self.health_tendency_counter + 1, 1)
        elif IS_BAD_HEALTH(health_score):
            self.health_tendency_counter = min(self.health_tendency_counter - 1, -1)
        else:
            self.health_tendency_counter = 0
        if self.health_tendency_counter >= TENDENCY_THRESHOLD:
            print(f"Queen Bee {self.id} is in good health. Reproducing.")
            self.presence_array = np.append(self.presence_array, 1)
            new_bee = Bee(self.id, no_bees)
            new_bee.set_spawn(self.spawn_location)
            self.action_space = MultiBinary(no_bees+1)
            return new_bee, True
        elif self.health_tendency_counter <= -TENDENCY_THRESHOLD:
            print(f"Queen Bee {self.id} is in bad health. Reducing population.")
            # sacrifices a bee, preferably if they're inside the beehive
            picked_index = self.presence_array.argmax()
            self.presence_array[picked_index] = 0
            return self.bees[picked_index], False
        return None, False

    def welcome(self, bee: Bee):
        self.presence_array[bee.local_beehive_id] = 1

