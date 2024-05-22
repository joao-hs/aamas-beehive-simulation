
import numpy as np


Coord = tuple[int, int]

class SearchingGuide:
    """
    Will hold the state necessary to perform a spreader search.
    Will commit to a direction and keep moving in that direction for a number of steps.
    If the agent hits an obstacle, it will change direction.
    """

    def __init__(self, moves, intent) -> None:
        """
        Moves should be like: [*_UP, *_DOWN, *_LEFT, *_RIGHT]
        """
        self.moves = moves
        self.intent = intent
        self.current_direction = None
        self.last_position = None
        self.steps = 0

    def walk(self, position: Coord) -> int:
        """
        Walk in the current direction. If the agent has walked the number of steps it intended to, it will change
        direction.
        """
        # change direction condition
        if self.last_position == position or self.steps == 0 or self.current_direction is None:
            self.current_direction = np.random.choice(self.moves)
            self.steps = self.intent
        else:
            self.steps -= 1
        self.last_position = position
        return self.current_direction
    