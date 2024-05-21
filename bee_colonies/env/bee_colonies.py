import functools
import random
from copy import copy
from os import environ

import numpy as np

from pettingzoo import ParallelEnv

from bee_colonies.models.flower import Flower

from bee_colonies.models.queen_bee import QueenBee
from bee_colonies.models.bee import Bee, BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, BEE_PICK, \
    BEE_DROP, BEE_N_ACTIONS
from bee_colonies.models.wasp import Wasp, WASP_STAY, WASP_UP, WASP_DOWN, WASP_LEFT, WASP_RIGHT, WASP_ATTACK, \
    WASP_N_ACTIONS
from bee_colonies.models.agent import Agent
from bee_colonies.models.grid import Grid


def configure_seed(seed):
    if seed is None:
        return
    environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


Coord = tuple[int, int]


class BeeColonyEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, queen_bees: list[QueenBee], bees: tuple[list[Bee], ...], wasps: list[Wasp], seed=None,
                 grid_shape=(64, 64),
                 n_bees_per_colony=(10,), flower_density=0.5, n_wasps=1, range_of_vision=2, max_steps=1000):
        """
        The init method takes in environment arguments.

        Should define the following attributes:
        - coordinates of each bee (dynamic)
        - coordinates of each flower (static)
        - coordinates of each beehive (static)
        - coordinates of each wasp (dynamic)
        - timestamp

        Default options:
        - seed: None
        - grid_shape: (64, 64) (size of the grid)
        - n_bees_per_colony: (10,) (number of bees per colony)
        - flower_density: 0.5 (probability of a flower being present in a cell)
        - max_steps: 1000
        """
        self._seed = seed
        configure_seed(self._seed)

        # Sizes
        self._grid_shape = grid_shape
        self._n_colonies = len(n_bees_per_colony)
        self._n_bees = sum(n_bees_per_colony)
        self._n_bees_per_colony = n_bees_per_colony
        self._n_wasps = n_wasps

        # Coordinates
        self.flower_coordinates: list[Coord] = None
        self.beehive_coordinates: list[Coord] = None
        self.bee_coordinates: tuple[list[Coord], ...] = None
        self.wasp_coordinates: list[Coord] = None

        # Agents
        self.queen_bees: list[QueenBee] = None
        self.bees_by_colony: tuple[list[Bee], ...] = None
        self.wasps: list[Wasp] = None

        # Masks
        self._range_of_vision = range_of_vision

        # Other attributes
        self.init_queen_bees = queen_bees
        self.init_bees = bees
        self.init_wasps = wasps

        self.flowers: dict[Coord, Flower] = None

        self.timestep: int = None
        self._flower_density = flower_density
        self._max_steps = max_steps
        self._grid = Grid(*self._grid_shape)

    def reset(self) -> tuple[list, tuple[list], list]:
        """
        Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - bee coordinates
        - wasp coordinates
        - flower coordinates
        - beehive coordinates
        - observationlocal_beehive_id
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """

        self.queen_bees: list[QueenBee] = copy(self.init_queen_bees)
        self.bees_by_colony: tuple[list[Bee], ...] = copy(self.init_bees)
        self.wasps: list[Wasp] = copy(self.init_wasps)
        self.timestep: int = 0

        self.flower_coordinates = [
            (a, b)
            for a in range(self._grid_shape[0])
            for b in range(self._grid_shape[1])
            if random.random() <= self._flower_density
        ]

        self.flowers = {
            flower_coord: Flower(flower_coord) for flower_coord in self.flower_coordinates
        }

        self.beehive_coordinates = []

        for _ in range(self._n_colonies):
            new_location = self.__assign_beehive_location()
            self.beehive_coordinates.append(new_location)

        self.bee_coordinates = tuple(
            [self.beehive_coordinates[bee.beehive_id] for bee in queen_bee.bees] for queen_bee in self.queen_bees
        )  # bees start at their respective beehives

        self.wasp_coordinates = [
            self.__assign_wasp_start_location() for _ in range(self._n_wasps)
        ]

        for queen_bee in self.queen_bees:
            queen_bee.set_spawn(self.beehive_coordinates[queen_bee.id])
            queen_bee.presence_array = np.ones(self._n_bees_per_colony[queen_bee.id])

        for colony in self.bees_by_colony:
            for bee in colony:
                bee.set_spawn(self.beehive_coordinates[bee.beehive_id])

        for wasp in self.wasps:
            wasp.set_spawn(self.wasp_coordinates[wasp.id])

        # Observation
        observations = (
            [self.__observation(agent) for agent in self.queen_bees],
            tuple(
                [self.__observation(bee) for bee in colony] for colony in self.bees_by_colony
            ),
            [self.__observation(agent) for agent in self.wasps]
        )

        return observations

    def step(self, actions: dict[Agent, int]):
        self.timestep += 1
        # TODO: rewards
        rewards = None
        # Execute actions
        for agent, action in actions.items():
            if agent.is_alive:
                self.__update_agent(agent, action)

        # Generate action masks
        # all can do all
        # go through each and exclude non-possible actions
        # bee and queen
        masks = self.init_masks()

        for queen_bee in self.queen_bees:
            for i, presence in enumerate(queen_bee.presence_array):
                if presence == 0:
                    masks[0][queen_bee.id][i] = 0
        for colony, colony_bees in enumerate(self.bees_by_colony):
            for bee in colony_bees:
                if not bee.is_alive:
                    # hack: put them in their beehive if they're dead
                    self.bee_coordinates[colony][bee.local_beehive_id] = bee.beehive_location
                    continue
                position: Coord = self.bee_coordinates[colony][bee.local_beehive_id]
                if position not in self.flower_coordinates:
                    masks[1][colony][bee.local_beehive_id][BEE_PICK] = 0
                if position not in self.wasp_coordinates:
                    masks[1][colony][bee.local_beehive_id][BEE_ATTACK] = 0
                if position == bee.beehive_location:
                    if self.queen_bees[bee.beehive_id].presence_array[bee.local_beehive_id] == 1:
                        # cannot move
                        masks[1][colony][bee.local_beehive_id] = np.zeros(bee.action_space.n, dtype=np.int8)
                        masks[1][colony][bee.local_beehive_id][BEE_STAY] = 1
                    else:
                        # needs to move out of the beehive
                        masks[1][colony][bee.local_beehive_id][BEE_STAY] = 0
        for wasp in self.wasps:
            position = self.wasp_coordinates[wasp.id]
            if position not in self.beehive_coordinates:
                masks[2][wasp.id][WASP_ATTACK] = 0

        # Check termination conditions
        done = self.timestep > self._max_steps or (
                all(not queen_bee.is_alive for queen_bee in self.queen_bees) and
                all(not bee.is_alive for colony in self.bees_by_colony for bee in colony)
        )

        # Get observations
        observations = (
            [self.__observation(agent) for agent in self.queen_bees],
            tuple(
                [self.__observation(bee) for bee in colony] for colony in self.bees_by_colony
            ),
            [self.__observation(agent) for agent in self.wasps]
        )

        # Infos
        infos = {}

        return observations, rewards, masks, done, infos

    def render(self):
        self._grid.populate(self.flowers, self.bee_coordinates, self.beehive_coordinates,
                            self.wasp_coordinates)
        self._grid.render()

    ## Helper functions

    def init_masks(self):
        return (
            [  # 2 in a multi binary action space means any action is possible
                2 * np.ones(queen_bee.action_space.n, dtype=np.int8) for queen_bee in self.queen_bees
            ],
            tuple(  # 1 in a discrete action space means the action is possible
                [np.ones(bee.action_space.n, dtype=np.int8) \
                 if bee.is_alive \
                 else np.zeros(bee.action_space.n, dtype=np.int8) \
                 for bee in colony] for colony in self.bees_by_colony
            ),
            [  # 1 in a discrete action space means the action is possible
                np.ones(wasp.action_space.n, dtype=np.int8) for wasp in self.wasps
            ]
        )

    def __random_position(self) -> Coord:
        return random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)

    def __taken_positions(self) -> list[Coord]:
        # agents do not occupy the space, they may overlap
        flower_coordinates = self.flower_coordinates or []
        beehive_coordinates = self.beehive_coordinates or []
        return flower_coordinates + beehive_coordinates

    def __random_available_position(self) -> Coord:
        position = self.__random_position()
        while position in self.__taken_positions():
            position = self.__random_position()
        return position

    def __assign_beehive_location(self) -> Coord:
        """Assigns a location for a new beehive, ensuring it is adequately spaced from existing beehives."""
        min_distance = 10  # Minimum acceptable distance between beehives, adjust as needed.

        while True:
            potential_location = self.__random_available_position()
            if all(self.__distance(potential_location, existing_location) >= min_distance for existing_location in
                   self.beehive_coordinates):
                return potential_location

    def __distance(self, pos1: Coord, pos2: Coord) -> float:
        """Calculates the Euclidean distance between two points."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def __assign_wasp_start_location(self) -> Coord:
        """Assigns a start location for a new wasp, ensuring it starts far from any beehives."""
        min_distance = 20  # Minimum distance from any beehive, adjust as needed.

        while True:
            potential_location = self.__random_available_position()
            if all(self.__distance(potential_location, beehive_location) >= min_distance for beehive_location in
                   self.beehive_coordinates):
                return potential_location

    def __wasp_at_position(self, position: Coord) -> int | None:
        """
        Check if a wasp is at the specified position.
        
        Parameters:
            position (tuple): The grid position to check for the presence of a wasp.

        Returns:
            int | None: The index of the wasp if a wasp is present at the position, otherwise None.
        """
        # Assuming wasp_coordinates stores tuples of wasp positions
        for index, wasp_pos in enumerate(self.wasp_coordinates):
            if wasp_pos == position:
                return index
        return None

    def __observation(self, agent: Agent):
        if isinstance(agent, QueenBee):
            center: Coord = agent.spawn_location
        elif isinstance(agent, Bee):
            center: Coord = self.bee_coordinates[agent.beehive_id][agent.local_beehive_id]
        elif isinstance(agent, Wasp):
            center: Coord = self.wasp_coordinates[agent.id]
        else:
            raise Exception("Unknown agent type")

        visible_cells = [
            (center[0] + i, center[1] + j)
            for i in range(-self._range_of_vision, self._range_of_vision + 1)
            for j in range(-self._range_of_vision, self._range_of_vision + 1)
            if 0 <= center[0] + i < self._grid_shape[0] and 0 <= center[1] + j < self._grid_shape[1]
        ]

        observation = {
            "beehives": [beehive_coord for beehive_coord in self.beehive_coordinates if beehive_coord in visible_cells],
            "flowers": [flower_coord for flower_coord in self.flower_coordinates if flower_coord in visible_cells],
            "bees": [bee_coord for bee_coord in self.bee_coordinates if bee_coord in visible_cells],
            "wasps": [wasp_coord for wasp_coord in self.wasp_coordinates if wasp_coord in visible_cells],
        }
        return observation

    def __update_agent(self, agent: Agent, action: int | np.ndarray):
        if not agent.is_alive:
            return
        if isinstance(agent, QueenBee):
            agent.presence_array = np.multiply(agent.presence_array, action)
            picked_bee, is_new = agent.timestep()
            if picked_bee:
                if is_new:
                    if picked_bee not in self.bees_by_colony[picked_bee.beehive_id]:
                        self.bees_by_colony[picked_bee.beehive_id].append(picked_bee)
                    self.bee_coordinates[picked_bee.beehive_id].append(picked_bee.beehive_location)
                else:
                    # self.bees_by_colony[picked_bee.beehive_id].remove(picked_bee)
                    picked_bee.is_alive = False
                    self.bee_coordinates[picked_bee.beehive_id][picked_bee.local_beehive_id] = picked_bee.beehive_location
        elif isinstance(agent, Bee):
            position: Coord = self.bee_coordinates[agent.beehive_id][agent.local_beehive_id]
            x, y = position
            if action == BEE_STAY:
                return
            elif action == BEE_UP:  # move up
                self.bee_coordinates[agent.beehive_id][agent.local_beehive_id] = self.__clamp_coord((x - 1, y))
            elif action == BEE_DOWN:  # move down
                self.bee_coordinates[agent.beehive_id][agent.local_beehive_id] = self.__clamp_coord((x + 1, y))
            elif action == BEE_LEFT:  # move left
                self.bee_coordinates[agent.beehive_id][agent.local_beehive_id] = self.__clamp_coord((x, y - 1))
            elif action == BEE_RIGHT:  # move right
                self.bee_coordinates[agent.beehive_id][agent.local_beehive_id] = self.__clamp_coord((x, y + 1))

            elif action == BEE_ATTACK:  # attack wasp
                for wasp in self.wasps:
                    wasp_position = self.wasp_coordinates[wasp.id]
                    if position == wasp_position:
                        wasp.receive_damage(agent.attack_power)
                        break

            elif action == BEE_PICK:  # pick up pollen
                if position not in self.flower_coordinates:
                    return
                if self.flowers[position].collect_pollen():
                    agent.collect_pollen()

            elif action == BEE_DROP:  # drop / enter beehive
                # Assuming you need to check if the bee is at its beehive location
                if position == agent.beehive_location:
                    queen_bee: QueenBee = self.queen_bees[agent.beehive_id]
                    if agent.drop_pollen():
                        queen_bee.receive_polen()
                    queen_bee.welcome(agent)
            else:
                raise Exception("Unknown action")

        elif isinstance(agent, Wasp):
            position: Coord = self.wasp_coordinates[agent.id]
            x, y = position
            if action == WASP_STAY:
                return
            elif action == WASP_UP:
                self.wasp_coordinates[agent.id] = self.__clamp_coord((x - 1, y))
            elif action == WASP_DOWN:
                self.wasp_coordinates[agent.id] = self.__clamp_coord((x + 1, y))
            elif action == WASP_LEFT:
                self.wasp_coordinates[agent.id] = self.__clamp_coord((x, y - 1))
            elif action == WASP_RIGHT:
                self.wasp_coordinates[agent.id] = self.__clamp_coord((x, y + 1))
            elif action == WASP_ATTACK:
                for queen_bee_id, beehive in enumerate(self.beehive_coordinates):
                    if position == beehive:
                        self.queen_bees[queen_bee_id].receive_damage(agent.attack_power)
                        break
            else:
                raise Exception("Unknown action")
        else:
            raise Exception("Unknown agent type")

    def __clamp_coord(self, coord):
        return max(0, min(coord[0], self._grid_shape[0] - 1)), max(0, min(coord[1], self._grid_shape[1] - 1))

    def __get_beehive_id(self, bee_number):
        for i, n_bees in enumerate(self._n_bees_per_colony):
            if bee_number < n_bees:
                return i
            bee_number -= n_bees
        return None

    def __get_local_beehive_id(self, bee_number):
        for _, n_bees in enumerate(self._n_bees_per_colony):
            if bee_number < n_bees:
                return bee_number
            bee_number -= n_bees
        return None
