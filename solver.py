from math import nan
from os import name, posix_spawn
import sys
import numpy as np
from numpy import Infinity, NaN, subtract, typing as npt

from dataclasses import dataclass

from queue import PriorityQueue

import numpy

import cProfile, pstats, io

import heapq

from tensorflow.python.framework.tensor import Tensor
from tqdm import tqdm

import tensorflow as tf
import keras

import sim


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def lineLine(
    a: tuple[tuple[float, float], tuple[float, float]],
    b: tuple[tuple[float, float], tuple[float, float]],
):
    # Thanks goes to jeffreythompson at jeffreythompson.org/collision-detection/line-line.php
    x1 = a[0][0]
    x2 = a[1][0]

    x3 = b[0][0]
    x4 = b[1][0]

    y1 = a[0][1]
    y2 = a[1][1]

    y3 = b[0][1]
    y4 = b[1][1]

    # print(f"l1(({x1}, {y1}), ({x2}, P{y2})), l2(({x3}, {y3}), ({x4}, P{y4}))")

    # calculate the distance to intersection point
    divisor = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    uA = NaN
    uB = NaN

    if divisor != 0.0:
        uA = numpy.divide(((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)), divisor)
        uB = numpy.divide(((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)), divisor)

    # if uA and uB are between 0-1, lines are colliding
    if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
        intersectionX = x1 + (uA * (x2 - x1))
        intersectionY = y1 + (uA * (y2 - y1))

        return True
    return False


def lineRect(
    rect: tuple[tuple[float, float], tuple[float, float]],
    line: tuple[tuple[float, float], tuple[float, float]],
):

    rx = rect[0][0]
    ry = rect[0][1]
    rw = rect[1][0]
    rh = rect[1][1]

    left = lineLine(line, ((rx, ry), (rx, ry + rh)))
    right = lineLine(line, ((rx + rw, ry), (rx + rw, ry + rh)))
    top = lineLine(line, ((rx, ry), (rx + rw, ry)))
    bottom = lineLine(line, ((rx, ry + rh), (rx + rw, ry + rh)))

    # print("collisionSummary:", (left, right, top, bottom))

    return left or right or top or bottom


@dataclass
class GridInfo:
    visited: bool
    isObstacle: bool
    shortestDist: np.float32
    optimalDirection: np.ndarray
    position: tuple[float, float]
    rc: tuple[int, int]
    solver: "Solver"
    valid: bool = True
    _priority: int | None = None

    def __eq__(self, other):
        return (
            self.visited == other.visited
            and self.isObstacle == other.isObstacle
            and self.shortestDist == other.shortestDist
            and (self.optimalDirection == other.optimalDirection).all()
            and self.position == other.position
            and self.rc == other.rc
        )

    def priority(self):
        if self._priority != None:
            return self._priority

        cnbrTraveledNeighbors = self.solver._validCellNeighbors(
            self.rc[0], self.rc[1], traveled=True
        )

        priorityFn = np.vectorize(lambda val: val.distanceTo(self) + val.distMag())

        self._priority = int(1000 * np.min(priorityFn(cnbrTraveledNeighbors))) - len(
            cnbrTraveledNeighbors
        )

        return self._priority

    def __lt__(self, other):
        return self.priority() < other.priority()

    def distMag(self) -> np.float32:
        return np.linalg.norm(self.shortestDist)

    def distTheta(self) -> np.float32:
        return np.arctan2(self.position[1], self.position[0])

    def displacementFrom(self, other):
        return np.subtract(self.position, other.position)

    def distanceTo(self, other):
        return np.linalg.norm(self.displacementFrom(other))

    def rect(self):
        return (
            (
                self.position[0] - self.solver.GRID_WIDTH / 2.0,
                self.position[1] - self.solver.GRID_HEIGHT / 2.0,
            ),
            (self.solver.GRID_WIDTH, self.solver.GRID_HEIGHT),
        )


class Solver:
    bufferCapacity = 100000
    batchSize = 64

    @dataclass
    class Observation:
        s: tuple[float, float, float, float] = tuple(
            np.zeros((4,)).tolist()
        )  # state = position, velocity
        a: tuple[float, float] = tuple(np.zeros(2).tolist())  # action = force
        r: float = 0.0  # reward = -distance
        sp: tuple[float, float, float, float] = tuple(
            np.zeros(4).tolist()
        )  # next state = next position, velocity

    def __init__(
        self,
        fieldWidth: float,
        fieldHeight: float,
        grid: np.ndarray,
        start: tuple[float, float],
        end: tuple[float, float],
        sim: sim.Sim,
    ):
        self.dataGrid = np.zeros(grid.shape, dtype=GridInfo)
        it = np.nditer(grid, flags=["multi_index"])
        for x in it:
            (i, j) = it.multi_index

            # if i % 2 == 0 or j % 2 == 0:
            #     continue
            # print((i, j))

            GRID_WIDTH = fieldWidth / grid.shape[1]
            GRID_HEIGHT = fieldHeight / grid.shape[0]
            self.GRID_WIDTH = GRID_WIDTH
            self.GRID_HEIGHT = GRID_HEIGHT

            self.dataGrid[i, j] = GridInfo(
                False,
                grid[i][j],
                np.float32(0.0),
                np.array((0.0, 0.0)),
                (
                    GRID_WIDTH * j + GRID_WIDTH / 2.0,
                    GRID_HEIGHT * i + GRID_HEIGHT / 2.0,
                ),
                (i, j),
                self,
            )

        self.fieldWidth = fieldWidth
        self.fieldHeight = fieldHeight

        self.start = start
        self.startGrid = self._gridFromIRL(start)

        self.end = end
        self.endGrid = self._gridFromIRL(end)

        self.neighborless = []
        self.sim = sim

        self.initializeShortestPaths()

        self.state_buffer = np.zeros((self.bufferCapacity, len(self.Observation().s)))
        self.action_buffer = np.zeros((self.bufferCapacity, len(self.Observation().a)))
        self.reward_buffer = np.zeros((self.bufferCapacity, 1))
        self.next_state_buffer = np.zeros(
            (self.bufferCapacity, len(self.Observation().sp))
        )
        self.observationCounter = 0

    def registerObservation(self, o: Observation):
        i = self.observationCounter % self.bufferCapacity

        self.state_buffer[i] = o.s
        self.action_buffer[i] = o.a
        self.reward_buffer[i] = o.r
        self.next_state_buffer[i] = o.sp

        self.observationCounter += 1

    def _gridFromIRL(self, pos: tuple[float, float]) -> tuple[int, int]:
        GRID_WIDTH = self.fieldWidth / self.dataGrid.shape[1]
        GRID_HEIGHT = self.fieldHeight / self.dataGrid.shape[0]

        return (
            (np.divide(pos, (GRID_WIDTH, GRID_HEIGHT)))
            .astype(dtype=int)
            .tolist()[0:2][::-1]
        )

    def _validCellNeighbors(
        self, parentRow: int, parentCol: int, includeObstacles=False, traveled=False
    ) -> list[GridInfo]:

        validNeighbors = []
        for row in range(parentRow - 1, parentRow + 2):
            for col in range(parentCol - 1, parentCol + 2):
                if (
                    row == parentRow
                    and col == parentCol
                    or row < 0
                    or row >= self.dataGrid.shape[0]
                    or col < 0
                    or col >= self.dataGrid.shape[1]
                ):
                    continue

                if includeObstacles != self.dataGrid[row][col].isObstacle:

                    # if np.array_equal((row, col), [36, 108]) and np.array_equal(
                    #     (parentRow, parentCol), [35, 108]
                    # ):
                    #     print("bad actor failed at obs")
                    continue

                if traveled != self.dataGrid[row][col].visited:
                    # if np.array_equal((row, col), [36, 108]) and np.array_equal(
                    #     (parentRow, parentCol), [35, 108]
                    # ):
                    # print(
                    #     "bad actor failed at trav",
                    #     traveled,
                    #     self.dataGrid[row][col].visited,
                    # )
                    continue

                if not self.dataGrid[row][col].valid:
                    # if np.array_equal((row, col), [36, 108]) and np.array_equal(
                    #     (parentRow, parentCol), [35, 108]
                    # ):
                    #     # print("bad actor failed at val")
                    continue

                validNeighbors.append(self.dataGrid[row][col])

                # if np.array_equal((row, col), [36, 108]) and np.array_equal(
                #     (parentRow, parentCol), [35, 108]
                # ):
                # print("reached bad actor parent")
        # print("validNeighborsLength:", len(validNeighbors))

        return validNeighbors

    def traverseForShortestPaths(self, pq: list, parentCell: GridInfo):
        traveledCellNeighbors = self._validCellNeighbors(
            parentCell.rc[0], parentCell.rc[1], traveled=True
        )

        def xwpm(nbr: GridInfo):
            xwpm_val = np.add(nbr.position, nbr.optimalDirection)
            # find the actual grid item that corresponds to xwpm_val
            xwpm_gridcoords = self._gridFromIRL((xwpm_val[0], xwpm_val[1]))

            xwpm_grid = self.dataGrid[xwpm_gridcoords[0]][xwpm_gridcoords[1]]

            return xwpm_grid

        def vk(xk: GridInfo, neighbors: list[GridInfo]):
            # print(f"endGrid: {self.endGrid}")
            # print(f"xk: {xk.rc}")
            if np.array_equal(self.endGrid, xk.rc):
                # print("same")
                return (np.float32(0.0), xk)

            def poss_vk(xk: GridInfo, nbr: GridInfo):

                xwpm_grid = xwpm(nbr)
                d = xwpm_grid.distanceTo(xk)

                return d + xwpm_grid.distMag()

            # Vk(Xk) eq (1)
            shortestDist_fn = np.vectorize(lambda nbr: poss_vk(xk, nbr))

            shortestDistances = shortestDist_fn(neighbors)
            idx = np.argmin(shortestDistances)

            return (shortestDistances[idx], neighbors[idx])

        def vd(xk: GridInfo, xwpm: GridInfo):
            return np.subtract(xwpm.position, xk.position)

        def verifyCollides(xk: GridInfo, nbr: GridInfo):
            xwpm_grid: GridInfo = xwpm(nbr)

            startRow = min(xwpm_grid.rc[0], xk.rc[0])
            endRow = max(xwpm_grid.rc[0], xk.rc[0])

            startCol = min(xwpm_grid.rc[1], xk.rc[1])
            endCol = max(xwpm_grid.rc[1], xk.rc[1])

            for row in range(startRow, endRow + 1):
                for col in range(startCol, endCol + 1):
                    targetGrid: GridInfo = self.dataGrid[row][col]

                    # print(
                    #     f"rect: {targetGrid.rect(), (xk.position, xwpm_grid.position)}"
                    # )
                    if targetGrid.isObstacle and lineRect(
                        targetGrid.rect(), (xk.position, xwpm_grid.position)
                    ):
                        return True
            return False

        traveledCellNeighborsBak = traveledCellNeighbors.copy()

        # print("preTraveledCell", traveledCellNeighbors)

        traveledCellNeighbors = [
            val for val in traveledCellNeighbors if not verifyCollides(parentCell, val)
        ]

        # if len(traveledCellNeighbors) == 0:
        #     traveledCellNeighbors = traveledCellNeighborsBak
        #
        #     for nbr in traveledCellNeighbors:

        traveledFN = np.vectorize(lambda val: val.visited)
        posFN = np.vectorize(lambda val: str(val.rc))

        # print("endGrid", self.endGrid)
        if len(traveledCellNeighbors) == 0 and not (
            np.array_equal(parentCell.rc, self.endGrid)
        ):

            def checkDirectCollision(a: GridInfo, b: GridInfo):
                startRow = min(a.rc[0], b.rc[0])
                endRow = max(a.rc[0], b.rc[0])

                startCol = min(a.rc[1], b.rc[1])
                endCol = max(a.rc[1], b.rc[1])

                for row in range(startRow, endRow + 1):
                    for col in range(startCol, endCol + 1):
                        targetGrid: GridInfo = self.dataGrid[row][col]
                        if targetGrid.isObstacle and lineRect(
                            targetGrid.rect(), (b.position, a.position)
                        ):
                            return True
                return False

            # just use the cell that we came from (or anything closer)
            traveledCellNeighbors = [
                val
                for val in traveledCellNeighborsBak
                if not checkDirectCollision(val, parentCell)
            ]

            fallBackCell = min(traveledCellNeighbors)

            parentCell.shortestDist = fallBackCell.shortestDist + np.linalg.norm(
                np.array(np.subtract(fallBackCell.position, parentCell.position))
            )
            parentCell.optimalDirection = np.subtract(
                fallBackCell.position, parentCell.position
            )

        else:
            (parentCell.shortestDist, optimalNbr) = vk(
                parentCell, traveledCellNeighbors
            )
            parentCell.optimalDirection = vd(parentCell, xwpm(optimalNbr))
        parentCell.visited = True

        untraveledCellNeighbors = self._validCellNeighbors(
            parentCell.rc[0], parentCell.rc[1], traveled=False
        )

        # print(traveledFN(self.dataGrid))

        # numpy.set_printoptions(threshold=sys.maxsize)
        # print(
        #     "gridState",
        #     traveledFN(
        #         self.dataGrid[
        #             parentCell.rc[0] - 1 : parentCell.rc[0] + 2,
        #             parentCell.rc[1] - 1 : parentCell.rc[1] + 2,
        #         ]
        #     ),
        # )
        # print(
        #     "gridPos",
        #     posFN(
        #         self.dataGrid[
        #             parentCell.rc[0] - 1 : parentCell.rc[0] + 2,
        #             parentCell.rc[1] - 1 : parentCell.rc[1] + 2,
        #         ]
        #     ),
        # )

        for cnbr in untraveledCellNeighbors:
            if np.array_equal(cnbr.rc, [35, 108]):
                # print("bad actor parent is", parentCell)
                pass
            # print(f"cnbr: {cnbr}")
            # print(f"parentTraveled: {self.dataGrid[self.endGrid[0]][self.endGrid[1]]}")
            cnbrTraveledNeighbors = self._validCellNeighbors(
                cnbr.rc[0], cnbr.rc[1], traveled=True
            )

            # verifyCollides_fn = np.vectorize(lambda val: verifyCollides(cnbr, val))
            #
            # if verifyCollides_fn(cnbrTraveledNeighbors).any():
            #     continue

            # if len(cnbrTraveledNeighbors) == 0:
            #     print("no neighbors:", cnbr)
            #     print("parent:", parentCell)

            if not (cnbr in pq):
                heapq.heappush(pq, cnbr)

    def initializeShortestPaths(self):
        # pr = cProfile.Profile()
        # pr.enable()

        queue = []
        heapq.heapify(queue)

        # Algorithm 1
        self.dataGrid[self.endGrid[0]][self.endGrid[1]].visited = True

        # 2.1 (Calculate the shortest distance from the goal to the center of the cell, set cell properties)
        # print(self.dataGrid[self.endGrid[0]][self.endGrid[1]])
        # print(self.endGrid[0], self.endGrid[1])

        # print(self.dataGrid[self.endGrid[1]][self.endGrid[0]].)
        print(f"start")

        self.traverseForShortestPaths(
            queue, self.dataGrid[self.endGrid[0]][self.endGrid[1]]
        )

        nonObstacleFN = np.vectorize(lambda val: not val.isObstacle)
        nNonObstacles = (nonObstacleFN(self.dataGrid) == True).sum()

        i = 0
        pbar = tqdm(total=nNonObstacles, ncols=100, bar_format="")
        prevTraveled = 0
        while len(queue) > 0:
            xk = heapq.heappop(queue)
            self.traverseForShortestPaths(queue, xk)
            i += 1

            traveledFN = np.vectorize(lambda val: val.visited)

            nTraveled = (traveledFN(self.dataGrid) == True).sum()
            pbar.update(int(nTraveled - prevTraveled))

            prevTraveled = nTraveled

        pbar.close()

        # pr.print_stats()
        # pr.disable()
        print(f"end")
        pass

    def naivePath(self, position: tuple[float, float]):
        startGrid = self._gridFromIRL(position)

        currentGrid = startGrid

        waypoints = [position]
        while not np.array_equal(currentGrid, self.endGrid):
            currentGridItem = self.dataGrid[currentGrid[0]][currentGrid[1]]
            nextPosition = np.add(
                currentGridItem.position, currentGridItem.optimalDirection
            )
            waypoints.append(nextPosition)

            currentGrid = self._gridFromIRL(nextPosition)

        return waypoints

    def createCritic(self):
        L = 10
        # input (sx, sy, vx, vy)
        # output (-deltaX, -deltaY)
        # return keras.Sequential([
        #     keras.layers.Dense(L, input_shape=(4,), activation=tf.nn.sigmoid),
        #     keras.layers.Dense(2),
        # ])

        print("input length:",len(self.Observation().s) + len(self.Observation().a), )

        inputs = keras.layers.Input(
            shape=(len(self.Observation().s) + len(self.Observation().a),)
        )
        hidden = keras.layers.Dense(L, activation="sigmoid", name="hidden2")(inputs)
        outputs = keras.layers.Dense(1, name="out")(hidden) * self.sim.maxF

        return keras.Model(inputs, outputs)

    def createActor(self):
        L = 10

        # input (sx, sy, vx, vy)
        # output (fx, fy)
        # return keras.Sequential(
        #     [
        #         keras.layers.Dense(
        #             L, input_shape=(len(self.Observation().s) + len(self.Observation().a),), activation=tf.nn.sigmoid
        #         ),
        #         keras.layers.Dense(2),
        #     ]
        # )

        inputs = keras.layers.Input(shape=(len(self.Observation().s),))
        hidden = keras.layers.Dense(L, activation="sigmoid", name="hidden")(inputs)
        outputs = keras.layers.Dense(2, name="out")(hidden) * self.sim.maxF

        return keras.Model(inputs, outputs)

    gamma = 0.99
    tau = 0.05

    critic_lr = 0.002
    actor_lr = 0.001

    @tf.function
    def update(self, state, action, reward, next_state):
        with tf.GradientTape() as tape:
            next_target_action = self.target_actor(next_state, training=True)
            y = reward + self.gamma * self.target_critic(
                np.concatenate(next_state, next_target_action), training=True
            )

            criticVal = self.critic(np.concatenate(state, action))
            L = keras.ops.mean(keras.ops.square(y - criticVal))

        gradient = tape.gradient(L, self.critic.trainable_weights)
        self.critic_optimizer.apply(zip(gradient, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            criticVal = self.critic(np.concatenate(state, action))
            L = keras.ops.mean(criticVal)

        gradient = tape.gradient(L, self.actor.trainable_weights)
        self.critic_optimizer.apply(zip(gradient, self.actor.trainable_weights))

    def elmWeightInitialization(self, model: keras.Model, XTrain, yTrain, nHidden):
        nInput = XTrain.shape[1]
        inputWeights = np.random.normal(size=[nInput, nHidden])
        biases = np.random.normal(size=[nHidden])

        def sigmoid(X):
            return 1 / (1 + np.exp(-X))

        def hiddenNodes(X):
            G = np.dot(X, inputWeights)
            G = G + biases
            H = sigmoid(G)

            return H

        outputWeights = np.dot(numpy.linalg.pinv(hiddenNodes(XTrain)), yTrain)

        print("inputweights", np.random.normal(size=[nInput, nHidden]).shape)
        print("hiddennodedim", numpy.linalg.pinv(hiddenNodes(XTrain)).shape)
        print("XDim", XTrain.shape)
        print("YDim", yTrain.shape)
        print("dim", outputWeights.shape)

        print(inputWeights)

        # model.layers[1].set_weights(inputWeights)
        # model.layers[2].set_weights(outputWeights)

        model.set_weights([
            [],
            inputWeights,
            outputWeights
        ])

    def generateActorELMData(self):
        X = []
        y = []

        # it = np.nditer(self.dataGrid, flags=["multi_index"])
        for r in range(self.dataGrid.shape[0]):
            for c in range(self.dataGrid.shape[1]):
                grid: GridInfo = self.dataGrid[r][c]

                if grid.isObstacle:
                    continue

                X.append(
                    np.array([
                        grid.position[0],
                        grid.position[1],
                        np.random.random() * 4.0 - 2.0,
                        np.random.random() * 4.0 - 2.0,
                    ])
                )

                if np.linalg.norm(grid.optimalDirection) == 0.0:
                    continue


                y.append(
                    np.multiply(
                        np.divide(
                            grid.optimalDirection, np.linalg.norm(grid.optimalDirection)
                        ),
                        self.sim.maxF,
                    )
                )

        return (np.array(X), np.array(y))

    def generateCriticELMData(self):
        X = []
        y = []

        for r in range(self.dataGrid.shape[0]):
            for c in range(self.dataGrid.shape[1]):
                grid: GridInfo = self.dataGrid[r][c]

                if grid.isObstacle:
                    continue

                if np.linalg.norm(grid.optimalDirection) == 0.0:
                    continue

                maxForceDirection = np.multiply(
                        np.divide(
                            grid.optimalDirection, np.linalg.norm(grid.optimalDirection)
                        ),
                        self.sim.maxF,
                    )

                X.append(
                    np.array([
                        grid.position[0],
                        grid.position[1],
                        numpy.sign(grid.optimalDirection[0]) * np.random.random() * 2.0,
                        numpy.sign(grid.optimalDirection[1]) * np.random.random() * 2.0,
                        maxForceDirection[0],
                        maxForceDirection[1]
                    ])
                )

                y.append(np.array([-grid.shortestDist]))

        return (np.array(X), np.array(y))

    def updateTarget(self, targetModel: keras.Model, model: keras.Model, tau: float):
        targetWeights = targetModel.get_weights()
        weights = model.get_weights()

        for i in range(len(targetWeights)):
            targetWeights[i] = targetWeights[i] * (1.0 - tau) + weights[i] * tau

    def learn(self):
        recordRange = min(self.observationCounter, self.bufferCapacity)

        batchIndices = np.random.choice(recordRange, self.batchSize)

        stateBatch = keras.ops.convert_to_tensor(self.state_buffer[batchIndices])
        actionBatch = keras.ops.convert_to_tensor(self.action_buffer[batchIndices])
        rewardBatch = keras.ops.convert_to_tensor(self.reward_buffer[batchIndices])
        nextStateBatch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batchIndices]
        )

        self.update(stateBatch, actionBatch, rewardBatch, nextStateBatch)

    def policy(self, state, noise):
        action = keras.ops.squeeze(self.actor(state)).numpy() + noise

        magActual = np.linalg.norm(action)

        magNew = np.clip(magActual, 0.0, self.sim.maxF)

        return (action / magActual) * magNew

    def reward(self, state, previousState):
        return -np.linalg.norm(
            np.subtract(
                (state[0], state[1]), (previousState[0], previousState[1])
            )
        )

    def nn(self):
        self.critic = self.createCritic()
        self.actor = self.createActor()

        for layer in self.critic.layers:
            print(layer.get_weights())

        self.target_critic = self.createCritic()
        self.target_actor = self.createActor()

        (actorX, actory) = self.generateActorELMData()
        (criticX, criticy) = self.generateCriticELMData()

        # self.elmWeightInitialization(self.critic, criticX, criticy, 10)
        # self.elmWeightInitialization(self.target_critic, criticX, criticy, 10)
        #
        # self.elmWeightInitialization(self.actor, actorX, actory, 10)
        # self.elmWeightInitialization(self.target_actor, actorX, actory, 10)

        self.critic_optimizer = keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(self.actor_lr)

        ouNoise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
        totalEpisodes = 1
        for ep in range(totalEpisodes):
            previousState = (
                self.sim.posX,
                self.sim.posY,
                self.sim.v * np.cos(self.sim.wheelTheta),
                self.sim.v * np.sin(self.sim.wheelTheta),
            )
            episodicReward = 0

            endGridCenter = (
                self.endGrid[1] * self.GRID_WIDTH + self.GRID_WIDTH / 2.0,
                self.endGrid[0] * self.GRID_HEIGHT + self.GRID_HEIGHT / 2.0,
            )

            # max 1000 iters
            for it in range(1000):
                # make it two dimentional
                tfPrevState = keras.ops.expand_dims(
                    keras.ops.convert_to_tensor(previousState), 0
                )

                action = self.policy(tfPrevState, ouNoise())

                self.sim.step(
                    float(np.linalg.norm((action[0], action[1]))),
                    float(np.arctan2(action[1], action[0])),
                )

                state = (
                    self.sim.posX,
                    self.sim.posY,
                    float(self.sim.v * np.cos(self.sim.wheelTheta)),
                    float(self.sim.v * np.sin(self.sim.wheelTheta)),
                )
                reward = self.reward(state, previousState)

                done = (
                    self.sim.posX == endGridCenter[0]
                    and self.sim.posY == endGridCenter[1]
                )

                self.registerObservation(
                    self.Observation(previousState, action, float(reward), state)
                )

                episodicReward += reward

                self.learn()

                self.updateTarget(self.target_actor, self.actor, self.tau)
                self.updateTarget(self.target_critic, self.actor, self.tau)

                if done:
                    break

                previousState = state
