import random
from copy import copy
from os import environ

import numpy as np

from pettingzoo import ParallelEnv

from bee_colonies.models.flower import Flower, generate_flowers, generate_uniform_flowers

from bee_colonies.models.queen_bee import HEALTH_SCORE_FUNCTION, QueenBee
from bee_colonies.models.bee import Bee, BEE_STAY, BEE_UP, BEE_DOWN, BEE_LEFT, BEE_RIGHT, BEE_ATTACK, BEE_PICK, \
    BEE_DROP, BEE_N_ACTIONS
from bee_colonies.models.wasp import Wasp, WASP_STAY, WASP_UP, WASP_DOWN, WASP_LEFT, WASP_RIGHT, WASP_ATTACK, \
    WASP_N_ACTIONS
from bee_colonies.models.agent import Agent, manhattan_distance
from bee_colonies.models.grid import Grid
from config import get_config

CONFIG = get_config()
QUEEN_BEE_VISION_MULTIPLIER = CONFIG["queen_bee_vision_multiplier"]
BEE_VISION_MULTIPLIER = CONFIG["bee_vision_multiplier"]
WASP_VISION_MULTIPLIER = CONFIG["wasp_vision_multiplier"]


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
                 grid_shape=(64, 64), n_bees_per_colony=(10,), flower_density=0.5, n_wasps=1, range_of_vision=2,
                 num_clusters=2, max_distance_from_cluster=5, section_size=5, max_steps=1000):
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
        self.seed = seed
        configure_seed(self.seed)

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
        self._num_clusters = num_clusters
        self._max_distance_from_cluster = max_distance_from_cluster
        self._section_size = section_size
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
        clusters = tuple()

        if self._num_clusters == 0:
            self.flower_coordinates = generate_uniform_flowers(self._grid_shape, self._flower_density)
        else:
            clusters = tuple(
                (np.random.randint(0, self._grid_shape[0]), np.random.randint(0, self._grid_shape[1]))
                for _ in range(self._num_clusters)
            )

            self.flower_coordinates = generate_flowers(self._grid_shape, self._flower_density, clusters)

        self.flowers = {
            flower_coord: Flower(flower_coord) for flower_coord in self.flower_coordinates
        }

        self.beehive_coordinates = []

        for _ in range(self._n_colonies):
            new_location = self.__assign_beehive_location(clusters)
            self.beehive_coordinates.append(new_location)

        self.bee_coordinates = tuple(
            [self.beehive_coordinates[bee.queen_id] for bee in queen_bee.bees] for queen_bee in self.queen_bees
        )  # bees start at their respective beehives

        self.wasp_coordinates = [
            self.__assign_wasp_start_location() for _ in range(self._n_wasps)
        ]

        for queen_bee in self.queen_bees:
            queen_bee.set_spawn(self.beehive_coordinates[queen_bee.id])
            queen_bee.presence_array = np.ones(self._n_bees_per_colony[queen_bee.id])

        for colony in self.bees_by_colony:
            for bee in colony:
                bee.set_spawn(self.beehive_coordinates[bee.queen_id])

        for wasp in self.wasps:
            wasp.set_spawn(self.wasp_coordinates[wasp.id])

        for queen_bee in self.queen_bees:
            for section in self.__get_all_sections():
                queen_bee.pursuing_flower_map[section] = set()
                queen_bee.section_size = self._section_size


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
        for flower in self.flowers.values():
            flower.timestep()
        # Execute actions
        for agent, action in actions.items():
            if agent.is_alive:
                self.__update_agent(agent, action)

        for flower in self.flowers.values():
            flower.timestep()

        # Generate action masks
        # all can do all
        # go through each and exclude non-possible actions
        # bee and queen
        masks = self.permissive_masks()

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
                wasp_at_position = self.__wasp_at_position(position)
                if wasp_at_position is None or not self.wasps[wasp_at_position].is_alive:
                    masks[1][colony][bee.local_beehive_id][BEE_ATTACK] = 0
                
                if position == bee.beehive_location:
                    # if position in self.wasp_coordinates:
                    #     masks[1][colony][bee.local_beehive_id][BEE_ATTACK] = 1
                    #     continue
                    if bee.pollen:
                        self.queen_bees[bee.queen_id].presence_array[bee.local_beehive_id] = 1
                        masks[1][colony][bee.local_beehive_id] = np.zeros(bee.action_space.n, dtype=np.int8)
                        masks[1][colony][bee.local_beehive_id][BEE_DROP] = 1
                    else:
                        if self.queen_bees[bee.queen_id].presence_array[bee.local_beehive_id] == 1:
                            # cannot move
                            masks[1][colony][bee.local_beehive_id] = np.zeros(bee.action_space.n, dtype=np.int8)
                            masks[1][colony][bee.local_beehive_id][BEE_STAY] = 1
                            masks[1][colony][bee.local_beehive_id][BEE_ATTACK] = 1
                        else:
                            # needs to move out of the beehive
                            masks[1][colony][bee.local_beehive_id][BEE_STAY] = 0
                    
        for wasp in self.wasps:
            if not wasp.is_alive:
                continue
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
        infos = {
            "timestep": self.timestep,
            "alive": {
                queen.id: queen.alive_bees for queen in self.queen_bees
            },
            "dead_count": {
                queen.id: [not bee.is_alive for bee in queen.bees].count(True) for queen in self.queen_bees
            },
            "food": {
                queen.id: queen.food_quantity for queen in self.queen_bees
            },
            "health": {
                queen.id: HEALTH_SCORE_FUNCTION(queen.food_quantity, queen.alive_bees) for queen in self.queen_bees
            },
            "health_tendency_counter": {
                queen.id: queen.health_tendency_counter for queen in self.queen_bees
            },
            "presence_in_beehive": {
                queen.id: np.count_nonzero(queen.presence_array == 1) for queen in self.queen_bees
            }
        }

        return observations, rewards, masks, done, infos

    def render(self):
        alive_wasps_coordinates = [
            wasp_coord for index, wasp_coord in enumerate(self.wasp_coordinates) if self.wasps[index].is_alive
        ]
        self._grid.populate(self.flowers, self.bee_coordinates, self.beehive_coordinates,
                            alive_wasps_coordinates)
        self._grid.render()

    ## Helper functions

    def permissive_masks(self):
        return (
            [  # 2 in a multi binary action space means any action is possible
                2 * np.ones(queen_bee.action_space.n, dtype=np.int8) \
                    if queen_bee.is_alive \
                    else np.zeros(queen_bee.action_space.n, dtype=np.int8) \
                for queen_bee in self.queen_bees
            ],
            tuple(  # 1 in a discrete action space means the action is possible
                [
                    np.ones(bee.action_space.n, dtype=np.int8) \
                        if bee.is_alive \
                        else np.zeros(bee.action_space.n, dtype=np.int8) \
                    for bee in colony
                ] for colony in self.bees_by_colony
            ),
            [  # 1 in a discrete action space means the action is possible
                np.ones(wasp.action_space.n, dtype=np.int8) \
                    if wasp.is_alive \
                    else np.zeros(wasp.action_space.n, dtype=np.int8) \
                for wasp in self.wasps
            ]
        )

    def init_masks(self):
        def no_move(n):
            mask = np.zeros(n, dtype=np.int8)
            mask[0] = 1  # assuming 0 means stay
            return mask

        return (
            [  # 2 in a multi binary action space means any action is possible
                2 * np.ones(queen_bee.action_space.n, dtype=np.int8) \
                for queen_bee in self.queen_bees
            ],
            tuple(
                [
                    no_move(bee.action_space.n) for bee in colony
                ] for colony in self.bees_by_colony
            ),
            [
                no_move(wasp.action_space.n) for wasp in self.wasps
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

    def __random_available_position_within(self, center, radius) -> Coord:
        bound_x = max(0, center[0] - radius), min(self._grid_shape[0] - 1, center[0] + radius)
        bound_y = max(0, center[1] - radius), min(self._grid_shape[1] - 1, center[1] + radius)
        possible_coords = [
            (x, y)
            for x in range(bound_x[0], bound_x[1] + 1)
            for y in range(bound_y[0], bound_y[1] + 1)
            if manhattan_distance((x, y), center) <= radius
        ]
        position = random.choice(possible_coords)
        while position in self.__taken_positions():
            position = random.choice(possible_coords)
        return position

    def __assign_beehive_location(self, clusters) -> Coord:
        """Assigns a location for a new beehive, ensuring it is adequately spaced from existing beehives."""
        min_distance = 10  # Minimum acceptable distance between beehives, adjust as needed.
        if clusters == tuple():
            all_coordinates = [(i, j) for i in range(self._grid_shape[0]) for j in range(self._grid_shape[1])]
            random.shuffle(all_coordinates)  # Shuffle to get a random order

            for potential_location in all_coordinates:
                if all(manhattan_distance(potential_location, existing_location) >= min_distance
                    for existing_location in self.beehive_coordinates):
                    return potential_location

            raise ValueError("No suitable location found for a new beehive")
        
        cluster = random.choice(clusters)

        while True:
            potential_location = self.__random_available_position_within(cluster, self._max_distance_from_cluster)
            if all(manhattan_distance(potential_location, existing_location) >= min_distance for existing_location in
                   self.beehive_coordinates):
                return potential_location

    def __assign_wasp_start_location(self) -> Coord:
        """Assigns a start location for a new wasp, ensuring it starts far from any beehives."""
        min_distance = 15  # Minimum distance from any beehive, adjust as needed.

        while True:
            potential_location = self.__random_available_position()
            if all(manhattan_distance(potential_location, beehive_location) >= min_distance for beehive_location in
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
        if not agent.is_alive:
            return self.__empty_obs()
        multiplier = 1
        if isinstance(agent, QueenBee):
            center: Coord = agent.spawn_location
            multiplier = QUEEN_BEE_VISION_MULTIPLIER
        elif isinstance(agent, Bee):
            center: Coord = self.bee_coordinates[agent.queen_id][agent.local_beehive_id]
            multiplier = BEE_VISION_MULTIPLIER
        elif isinstance(agent, Wasp):
            center: Coord = self.wasp_coordinates[agent.id]
            multiplier = WASP_VISION_MULTIPLIER
        else:
            raise Exception("Unknown agent type")

        visible_cells = [
            (center[0] + i, center[1] + j)
            for i in range(-int(self._range_of_vision * multiplier), int(self._range_of_vision * multiplier) + 1)
            for j in range(-int(self._range_of_vision * multiplier), int(self._range_of_vision * multiplier) + 1)
            if 0 <= center[0] + i < self._grid_shape[0] and 0 <= center[1] + j < self._grid_shape[1]
        ]

        observation = {
            "position": center,
            "beehives": [ 
                        (beehive_coord, self.queen_bees[index].is_alive)  
                        for index, beehive_coord in enumerate(self.beehive_coordinates)
                        if beehive_coord in visible_cells
                        ],
            "flowers": [flower for flower in self.flowers.values() if flower.position in visible_cells],
            "bees": [
                (colony, i, bee_coord)
                for colony, colony_coords in enumerate(self.bee_coordinates)
                for i, bee_coord in enumerate(colony_coords) if bee_coord in visible_cells
            ],
            "wasps": [(wasp_coord, self.wasps[index].is_alive) for index, wasp_coord in enumerate(self.wasp_coordinates) if wasp_coord in visible_cells],
        }
        return observation

    def __empty_obs(self):
        return {
            "position": None,
            "beehives": [],
            "flowers": [],
            "bees": [],
            "wasps": [],
        }

    def __update_agent(self, agent: Agent, action: int | np.ndarray):
        if not agent.is_alive:
            return
        if isinstance(agent, QueenBee):
            agent.presence_array = np.multiply(agent.presence_array, action)
            picked_bee, is_new = agent.timestep()
            if picked_bee:
                if is_new:
                    if picked_bee not in self.bees_by_colony[picked_bee.queen_id]:
                        self.bees_by_colony[picked_bee.queen_id].append(picked_bee)
                    self.bee_coordinates[picked_bee.queen_id].append(picked_bee.beehive_location)
                else:
                    # self.bees_by_colony[picked_bee.beehive_id].remove(picked_bee)
                    picked_bee.is_alive = False
                    self.bee_coordinates[picked_bee.queen_id][picked_bee.local_beehive_id] = None
                    # queen.dead_bee(...) is called on timestep(), do not call it here
        elif isinstance(agent, Bee):
            position: Coord = self.bee_coordinates[agent.queen_id][agent.local_beehive_id]
            x, y = position
            if action == BEE_STAY:
                return
            elif action == BEE_UP:  # move up
                self.bee_coordinates[agent.queen_id][agent.local_beehive_id] = self.__clamp_coord((x - 1, y))
            elif action == BEE_DOWN:  # move down
                self.bee_coordinates[agent.queen_id][agent.local_beehive_id] = self.__clamp_coord((x + 1, y))
            elif action == BEE_LEFT:  # move left
                self.bee_coordinates[agent.queen_id][agent.local_beehive_id] = self.__clamp_coord((x, y - 1))
            elif action == BEE_RIGHT:  # move right
                self.bee_coordinates[agent.queen_id][agent.local_beehive_id] = self.__clamp_coord((x, y + 1))

            elif action == BEE_ATTACK:  # attack wasp
                for wasp in self.wasps:
                    if not wasp.is_alive:
                        continue
                    wasp_position = self.wasp_coordinates[wasp.id]
                    if position == wasp_position:
                        if wasp.health > 0:
                            wasp.receive_damage(agent.attack_power)
                            agent.is_alive = False  # kamikaze
                            agent.queen.dead_bee(agent.local_beehive_id)
                            # no need to move to beehive since it's already there
                        else:
                            wasp.is_alive = False
                            
                        break

            elif action == BEE_PICK:  # pick up pollen
                if position not in self.flower_coordinates:
                    return
                if self.flowers[position].collect_pollen():
                    agent.collect_pollen()

            elif action == BEE_DROP:  # drop / enter beehive
                # Assuming you need to check if the bee is at its beehive location
                if position == agent.beehive_location:
                    queen_bee: QueenBee = self.queen_bees[agent.queen_id]
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
                        if self.queen_bees[queen_bee_id].is_alive:
                            self.queen_bees[queen_bee_id].receive_damage(agent.attack_power)
            else:
                raise Exception("Unknown action")
        else:
            raise Exception("Unknown agent type")
        
    def __find_new_position_after_attack(self, wasp_id):
        # Simple strategy: move one step in a random direction, ensuring it's within bounds
        current_position = self.wasp_coordinates[wasp_id]
        possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random_move = random.choice(possible_moves)
        new_x = max(0, min(self._grid_shape[0] - 1, current_position[0] + random_move[0]))
        new_y = max(0, min(self._grid_shape[1] - 1, current_position[1] + random_move[1]))
        return (new_x, new_y)

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

    def __get_all_sections(self):
        return [
            (a, b)
            for a in range(0, self._grid_shape[0], self._section_size)
            for b in range(0, self._grid_shape[1], self._section_size)
        ]

