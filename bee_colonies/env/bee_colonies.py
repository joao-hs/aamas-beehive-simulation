import functools
import random
from copy import copy
from os import environ

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary

from pettingzoo import ParallelEnv

from ..models.queen_bee import QueenBee
from ..models.bee import Bee
from ..models.wasp import Wasp
from ..models.agent import Agent
from ..models.grid import Grid

def configure_seed(seed):
    if seed is None:
        return
    environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

BEE_POSSIBLE_ACTIONS = range(6) # 0: stay still, 1: move up, 2: move down, 3: move left, 4: move right, 5: interact
WASP_POSSIBLE_ACTIONS = range(6) # 0: stay still, 1: move up, 2: move down, 3: move left, 4: move right, 5: interact
QUEENBEE_POSSIBLE_ACTIONS = None # depends on the number of bees in the colony. For each bee, decide 0: let go, 1: keep

class BeeColonyEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    agents: list[Agent]

    def __init__(self, seed=None, grid_shape=(64, 64), n_bees_per_colony=(10,), flower_density=0.5, n_wasps=1, range_of_vision=2, max_steps=1000):
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
        self.flower_coordinates = None
        self.beehive_coordinates = None
        self.bee_coordinates = None
        self.wasp_coordinates = None
        self.all_agents_coordinates = None

        # Masks
        self._range_of_vision = range_of_vision

        # Other attributes
        self.initial_agents_state = [
            QueenBee(beehive_id, self._n_bees_per_colony[beehive_id]) for beehive_id in range(self._n_colonies)
        ] + [
            Bee(self._n_colonies + bee_id, self.__get_beehive_id(bee_id), self.__get_local_beehive_id(bee_id)) for bee_id in range(self._n_bees)
        ] + [
            Wasp(self._n_colonies + self._n_bees + wasp_id) for wasp_id in range(self._n_wasps)
        ]

        self.timestep = None
        self._flower_density = flower_density
        self._max_steps = max_steps
        self._grid = Grid(*self._grid_shape)

    def reset(self):
        """
        Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - bee coordinates
        - wasp coordinates
        - flower coordinates
        - beehive coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        # TODO: should reset seed?

        self.agents = copy(self.initial_agents_state)
        self.timestep = 0

        self.flower_coordinates = [
            (a,b) for a in range(self._grid_shape[0]) for b in range(self._grid_shape[1]) if random.random() <= self._flower_density
        ]

        self.beehive_coordinates = [
            self.__assign_beehive_location() for _ in range(self._n_colonies)
        ]
        
        self.bee_coordinates = [
            self.beehive_coordinates[colony] for colony in range(self._n_colonies) for _ in range(self._n_bees_per_colony[colony])
        ] # bees start at their respective beehives

        self.wasp_coordinates = [
           self.__assign_wasp_start_location() for _ in range(self._n_wasps)
        ]
        
        for agent in self.agents:
            if isinstance(agent, QueenBee):
                agent.set_spawn(self.beehive_coordinates[agent.id])
                agent.presence_array = np.ones(self._n_bees_per_colony[agent.id])
            elif isinstance(agent, Bee):
                agent.set_spawn(self.bee_coordinates[agent.id - self._n_colonies])
            elif isinstance(agent, Wasp):
                agent.set_spawn(self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees])


        # Observation
        observations = [
            self.__observation(agent) for agent in self.agents
        ]

        # Define observation and action spaces
        self.observation_spaces = {
            agent.id: MultiDiscrete([
                2 * self._range_of_vision + 1, 2 * self._range_of_vision + 1, 2, 2, 2, 2
            ]) if not isinstance(agent, QueenBee) else Discrete(1) for agent in self.agents
        }

        self.action_spaces = {
            # each bee can move in 4 directions, stay still or interact with the current position
            # the queenbee has to decide if it will let go a bee or not
            agent.id: Discrete(6) if isinstance(agent, Bee) or isinstance(agent, Wasp) else MultiBinary(self._n_bees_per_colony[agent.id]) for agent in self.agents
        }

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
        
        masks = {
            agent.id: np.ones(self.action_spaces[agent.id].n, dtype=np.int8) if not isinstance(agent, QueenBee) else 2*np.ones(self.action_spaces[agent.id].n, dtype=np.int8) for agent in self.agents
        }

        for agent in self.agents:
            if isinstance(agent, Bee):
                position = self.bee_coordinates[agent.id - self._n_colonies]
                if position != agent.beehive_location:
                    self.agents[agent.beehive_id].presence_array[agent.local_beehive_id] = 0
                else:
                    if self.agents[agent.beehive_id].presence_array[agent.local_beehive_id] == 1: # inside the beehive (and queen is account for it)
                        # cannot move
                        masks[agent.id] = np.zeros(self.action_spaces[agent.id].n, dtype=np.int8)
                        masks[agent.id][0] = 1
                    else:
                        # needs to move out of the beehive
                        masks[agent.id][0] = 0
            if isinstance(agent, QueenBee):
                for i, presence in enumerate(agent.presence_array):
                    if presence == 0:
                        masks[agent.id][i] = 0

        # Check termination conditions
        done = self.timestep >= self._max_steps or all(not agent.is_alive for agent in self.agents)

        # Get observations
        observations = [
            self.__observation(agent) for agent in self.agents
        ]

        # Infos
        infos = {}
        
        return observations, rewards, masks, done, infos
        

    def render(self):
        self._grid.populate(self.flower_coordinates, self.bee_coordinates, self.beehive_coordinates, self.wasp_coordinates)
        self._grid.render()
        

    def observation_space(self, agent_name):
        return self.observation_spaces[agent_name]

    def action_space(self, agent_name):
        return self.action_spaces[agent_name]
    

    ## Helper functions

    def __random_position(self) -> tuple[int]:
        return (random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1))

    def __taken_positions(self) -> list[tuple[int]]:
        # agents do not occupy the space, they may overlap
        flower_coordinates = self.flower_coordinates or []
        beehive_coordinates = self.beehive_coordinates or []
        return flower_coordinates + beehive_coordinates

    def __random_available_position(self) -> tuple[int]:
        position = self.__random_position()
        while position in self.__taken_positions():
            position = self.__random_position()
        return position

    def __assign_beehive_location(self) -> tuple[int]:
        # TODO: ensure that beehives are scattered enough
        return self.__random_available_position()
    
    def __assign_wasp_start_location(self) -> tuple[int]:
        # TODO: probably should start somewhere far from the beehives
        return self.__random_available_position()
    
    def __observation(self, agent: Agent):
        if isinstance(agent, QueenBee):
            center = agent.spawn_location
        elif isinstance(agent, Bee):
            center = self.bee_coordinates[agent.id - self._n_colonies]
        elif isinstance(agent, Wasp):
            center = self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees]
        else:
            raise Exception("Unknown agent type")

        visible_cells = [
            (center[0] + i, center[1] + j) for i in range(-self._range_of_vision, self._range_of_vision + 1) for j in range(-self._range_of_vision, self._range_of_vision + 1) if 0 <= center[0] + i < self._grid_shape[0] and 0 <= center[1] + j < self._grid_shape[1]
        ]

        observation = {
            "spawn": agent.spawn_location,
            "beehives": [beehive_coord for beehive_coord in self.beehive_coordinates if beehive_coord in visible_cells],
            "flowers": [flower_coord for flower_coord in self.flower_coordinates if flower_coord in visible_cells],
            "bees": [bee_coord for bee_coord in self.bee_coordinates if bee_coord in visible_cells],
            "wasps": [wasp_coord for wasp_coord in self.wasp_coordinates if wasp_coord in visible_cells],
        }
        return observation

    def __update_agent(self, agent: Agent, action: int | np.ndarray):
        if isinstance(agent, QueenBee):
            agent.presence_array = np.multiply(agent.presence_array, action)
        elif isinstance(agent, Bee):
            x, y = self.bee_coordinates[agent.id - self._n_colonies]
            if action == 0:
                return
            elif action == 1: # move up
                self.bee_coordinates[agent.id - self._n_colonies] = self.__clamp_coord((x - 1, y))
            elif action == 2: # move down
                self.bee_coordinates[agent.id - self._n_colonies] = self.__clamp_coord((x + 1, y))
            elif action == 3: # move left
                self.bee_coordinates[agent.id - self._n_colonies] = self.__clamp_coord((x, y - 1))
            elif action == 4: # move right
                self.bee_coordinates[agent.id - self._n_colonies] = self.__clamp_coord((x, y + 1))
            elif action == 5: # interact
                position = self.bee_coordinates[agent.id - self._n_colonies]
                if position in self.flower_coordinates:
                    # consume flower
                    pass
                elif position in self.beehive_coordinates:
                    # return to beehive
                    pass
                elif position in self.wasp_coordinates:
                    # attack wasp
                    pass
            else:
                raise Exception("Unknown action")

        elif isinstance(agent, Wasp):
            x, y = self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees]
            if action == 0:
                return
            elif action == 1:
                self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees] = self.__clamp_coord((x - 1, y))
            elif action == 2:
                self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees] = self.__clamp_coord((x + 1, y))
            elif action == 3:
                self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees] = self.__clamp_coord((x, y - 1))
            elif action == 4:
                self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees] = self.__clamp_coord((x, y + 1))
            elif action == 5:
                position = self.wasp_coordinates[agent.id - self._n_colonies - self._n_bees]
                if position in self.beehive_coordinates:
                    # attack beehive
                    pass
            else:
                raise Exception("Unknown action")
            # TODO check if the wasp is on a beehive

        else:
            raise Exception("Unknown agent type")

    def __clamp_coord(self, coord):
        return (max(0, min(coord[0], self._grid_shape[0] - 1)), max(0, min(coord[1], self._grid_shape[1] - 1)))
    
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

