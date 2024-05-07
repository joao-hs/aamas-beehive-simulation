import functools
import random
from copy import copy
from os import environ

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

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
            QueenBee(beehive_id) for beehive_id in range(self._n_colonies)
        ] + [
            Bee(self._n_colonies + bee_id) for bee_id in range(self._n_bees)
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
            # each bee can move in 4 directions or stay still
            # the queenbee only has two actions: let go N bees or do nothing
            agent.id: Discrete(5) if isinstance(agent, Bee) or isinstance(agent, Wasp) else Discrete(2) for agent in self.agents
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
        # TODO: array of 1's the same size as the number of possible actions
        # TODO: 0 on the impossible actions

        # Check termination conditions
        done = self.timestep >= self._max_steps or all(not agent.is_alive for agent in self.agents)

        # Get observations
        observations = [
            self.__observation(agent) for agent in self.agents
        ]

        # Infos
        infos = {}
        
        return observations, rewards, done, infos
        

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

    def __update_agent(self, agent: Agent, action: int):
        if isinstance(agent, QueenBee):
            # TODO
            return
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
            else:
                raise Exception("Unknown action")
            # TODO: check if the bee is on a flower

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
            else:
                raise Exception("Unknown action")
            # TODO check if the wasp is on a beehive

        else:
            raise Exception("Unknown agent type")

    def __clamp_coord(self, coord):
        return (max(0, min(coord[0], self._grid_shape[0] - 1)), max(0, min(coord[1], self._grid_shape[1] - 1)))


