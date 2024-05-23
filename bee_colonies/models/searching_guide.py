from copy import copy

import numpy as np

from bee_colonies.models.agent import manhattan_distance

Coord = tuple[int, int]


class SearchingGuide:
    """
    Will hold the state necessary to perform a spreader search.
    Will commit to a direction and keep moving in that direction for a number of steps.
    If the agent hits an obstacle, it will change direction.
    """

    def __init__(self, moves, intent, cluster_center_distance) -> None:
        """
        Moves should be like: [*_UP, *_DOWN, *_LEFT, *_RIGHT]
        """
        self.moves = moves
        self.UP, self.DOWN, self.LEFT, self.RIGHT = self.moves
        self.intent = intent
        self.current_direction = None
        self.last_position = None
        self.steps = 0
        self.seen_objects = set()
        self.expected_clusters = list()
        self.cluster_center_distance = cluster_center_distance

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

    def walk_to_cluster(self, position: Coord) -> int:
        print(self.expected_clusters)
        self.expected_clusters.sort(key=lambda x: manhattan_distance(position, x))

        if not self.expected_clusters:
            # no clusters found yet, randomly walking
            return self.walk(position)

        closer_cluster = self.expected_clusters[0]
        x1, y1 = position
        x2, y2 = closer_cluster

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) + abs(dy) < self.cluster_center_distance:
            # already in cluster, walk randomly
            return self.walk(position)

        if dx > dy:
            return self.DOWN if x2 < x1 else self.UP
        else:
            return self.RIGHT if y2 < y1 else self.LEFT

    def share(self):
        return copy(self.seen_objects)

    def retrieve(self, objects):
        self.seen_objects.update(objects)

    def set_clusters(self, clusters: np.array):
        if clusters is None:
            return
        self.expected_clusters = copy(clusters.tolist())
