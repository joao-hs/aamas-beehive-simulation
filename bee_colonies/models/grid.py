import numpy as np
import pygame as pg

# dark green
BACKGROUND_COLOR = (0, 100, 0)

COLORS = {
    "F": (255, 182, 193),  # pink
    "F'": (110, 58, 60),  # brown
    "H": (255, 0, 0),  # red
    "W": (255, 140, 0),  # orange
    "B": (255, 255, 0)  # yellow
}

TICK_RATE = 60


class Grid:
    def __init__(self, width, height):
        self.empty = np.array([[" " for _ in range(width)] for _ in range(height)])
        self.grid = np.array([[" " for _ in range(width)] for _ in range(height)])
        self.uwidth = width
        self.uheight = height
        self.screen_size = (600, 600)
        self.cell_size = min(600 // width, 600 // height)
        self.clock = pg.time.Clock()
        pg.init()
        self.screen = pg.display.set_mode(self.screen_size)
        pg.display.set_caption("Bee Colonies")
        self.screen.fill(BACKGROUND_COLOR)

    def populate(self, flowers, bees_by_colonies, beehives, wasps):
        self.grid = self.empty.copy()
        for flower_position, flower in flowers.items():
            self.grid[flower_position[0], flower_position[1]] = "F" if flower.pollen else "F'"
        for colony in bees_by_colonies:
            for bee in colony:
                self.grid[bee[0], bee[1]] = "B"
        for beehive in beehives:
            self.grid[beehive[0], beehive[1]] = "H"
        for wasp in wasps:
            self.grid[wasp[0], wasp[1]] = "W"

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        for i in range(self.uwidth):
            for j in range(self.uheight):
                if self.grid[j][i] != " ":
                    pg.draw.rect(self.screen, COLORS[self.grid[j][i]],
                                 (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size))
        pg.display.update()
        # with np.printoptions(threshold=np.inf):
        #     for row in self.grid:
        #         print(" ".join(row))
        self.clock.tick(TICK_RATE)
