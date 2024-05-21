import numpy as np
from abc import ABC, abstractmethod

Coord = tuple[int, int]


def apply_mask_to_action(action, mask) -> int | np.ndarray:
    if mask is None:
        return action
    if isinstance(action, int):
        return action * mask[action]
    elif isinstance(action, np.ndarray):
        for m, (index, _) in zip(mask, enumerate(action)):
            if m == 0 or m == 1:
                action[index] = m
    return action


def manhattan_distance(coord1: Coord, coord2: Coord):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


class Agent(ABC):
    """
    Base agent class.
    Represents the concept of an autonomous agent.

    Attributes
    ----------
    name: str
        Name for identification purposes.
        
    observation: np.ndarray
       The most recent observation of the environment


    Methods
    -------
    see(observation)
        Collects an observation

    action(): int
        Abstract method.
        Returns an action, represented by an integer
        May take into account the observation (numpy.ndarray).

    References
    ----------
    ..[1] Michael Wooldridge "An Introduction to MultiAgent Systems - Second
    Edition", John Wiley & Sons, p 44.


    """

    def __init__(self):
        self.spawn_location: Coord = None
        self.last_observation = None
        self.mask = None
        self.is_alive = None
        self.action_space = None

    def set_spawn(self, spawn_location: Coord):
        self.spawn_location = spawn_location

    def see(self, observation: np.ndarray, mask: np.ndarray = None):
        self.last_observation = observation
        self.mask = mask

    @abstractmethod
    def action(self) -> int | np.ndarray:
        raise NotImplementedError()
    
    def distance(self, pos1, pos2) -> int:
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1-x2) + abs(y1-y2)