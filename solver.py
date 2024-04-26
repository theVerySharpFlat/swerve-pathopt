import numpy as np
from numpy import typing as npt

from dataclasses import dataclass

from queue import PriorityQueue


@dataclass
class GridInfo:
    visited: bool
    isObstacle: bool
    shortestDist: np.ndarray
    position: tuple[float, float]

    def distMag(self) -> np.float32:
        return np.linalg.norm(self.shortestDist)

    def distTheta(self) -> np.float32:
        return np.arctan2(self.position[1], self.position[0])

    def distanceFrom(self, other):
        pass


class Solver:
    def __init__(
        self,
        fieldWidth: float,
        fieldHeight: float,
        grid: np.ndarray,
        start: tuple[float, float],
        end: tuple[float, float],
    ):
        self.dataGrid = np.zeros(grid.shape, dtype=GridInfo)
        it = np.nditer(grid, flags=["multi_index"])
        for x in it:
            (i, j) = it.multi_index

            GRID_WIDTH = fieldWidth / grid.shape[0]
            GRID_HEIGHT = fieldHeight / grid.shape[1]

            self.dataGrid[i, j] = GridInfo(
                False,
                grid[i][j],
                np.array((0, 0)),
                (
                    GRID_WIDTH * i + GRID_WIDTH / 2.0,
                    GRID_HEIGHT * j + GRID_HEIGHT / 2.0,
                ),
            )

    def initializeShortestPaths(self):
        queue = PriorityQueue()
        pass
