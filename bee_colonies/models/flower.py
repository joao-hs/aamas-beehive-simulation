import numpy as np

TIME_TO_RESTORE_POLLEN = 5
SPREAD_DIVIDER = 7
SPREAD_SCALE = 0.1

Coord = tuple[int, int]

class Flower:
    def __init__(self, position) -> None:
        self.pollen = True
        self.position = position
        self.counter = 0

    def collect_pollen(self) -> bool:
        if self.pollen:
            self.pollen = False
            return True
        return False

    def timestep(self):
        if self.pollen:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= TIME_TO_RESTORE_POLLEN:
                self.pollen = True
                self.counter = 0

    def __repr__(self):
        return f"{self.position[0]},{self.position[1]}{'*' if self.pollen else ''}"

def generate_flowers(grid_shape: Coord, flower_density: float, hotspots: tuple[Coord, ...]) -> list[Coord]:
    flower_coordinates = set()
    num_hotspots = len(hotspots)
    max_flowers_per_hotspot = int((grid_shape[0] * grid_shape[1] * (flower_density + 0.05)) / num_hotspots)
    min_flower_per_hotspot = int((grid_shape[0] * grid_shape[1] * (flower_density - 0.05)) / num_hotspots)
    for center in hotspots:
        spread = np.random.normal(grid_shape[0] // SPREAD_DIVIDER, SPREAD_SCALE)
        num_flowers = np.random.randint(min_flower_per_hotspot, max_flowers_per_hotspot)
        for _ in range(num_flowers):
            flower_coord = int(np.random.normal(center[0], spread)), \
               int(np.random.normal(center[1], spread))
            if 0 <= flower_coord[0] < grid_shape[0] and 0 <= flower_coord[1] < grid_shape[1]:
                flower_coordinates.add(flower_coord)
    return list(flower_coordinates)



