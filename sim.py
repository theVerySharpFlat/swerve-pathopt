import math
import sys
import numpy
import pygame


def collidePointRect(
    r1: tuple[tuple[float, float], tuple[float, float]],
    p: tuple[float, float],
    inclusive: bool = True
):
    r1x = round(r1[0][0], 8)
    r1y = round(r1[0][1], 8)
    r1w = round(r1[1][0], 8)
    r1h = round(r1[1][1], 8)

    if inclusive:
        return p[0] >= r1x and p[0] <= r1x + r1w and p[1] >= r1y and p[1] <= r1y + r1h
    else:
        return p[0] > r1x and p[0] < r1x + r1w and p[1] > r1y and p[1] < r1y + r1h



class Sim:
    def __init__(
        self,
        posX: float,
        posY: float,
        deltaT: float,
        mass: float,
        muK: float,
        length: float,
        WINDOW,
    ):
        self.posX = posX
        self.posY = posY
        self.v = 0.0
        self.mass = mass
        self.wheelTheta = 0.0
        self.deltaT = deltaT
        self.muK = muK
        self.length = length
        self.WINDOW = WINDOW

    def _friction(self, vel):
        return -numpy.sign(vel) * 9.81 * self.mass * self.muK

    def _forceSim(self, F: float, tMax: float) -> float:
        """Returns total time took for this iteration"""

        Ff = self._friction(self.v)
        if self.v == 0:
            friction = self._friction(F)
            # friction in direction of F
            if numpy.abs(Ff) >= numpy.abs(F):
                return tMax

        # print(f"friction: {Ff}")

        # compute next time when v = 0
        zeroT = numpy.divide(numpy.multiply(-self.v, self.mass), numpy.add(Ff, F))

        endTime = zeroT if (zeroT > 0 and zeroT < tMax) else tMax

        a = numpy.divide(numpy.add(F, Ff), self.mass)
        D = numpy.multiply(self.v, endTime) + 0.5 * numpy.multiply(
            a, numpy.power(endTime, 2)
        )
        # print(f"self.v = {self.v} + {a} * {endTime}")
        self.v = self.v + a * endTime

        self.posX += D * numpy.cos(self.wheelTheta)
        self.posY += D * numpy.sin(self.wheelTheta)

        return endTime

    def toRect(self):
        return (
            (self.posX - self.length / 2.0, self.posY - self.length / 2.0),
            (self.length, self.length),
        )

    def _resolveCollision(
        self, rect: tuple[tuple[float, float], tuple[float, float]]
    ) -> tuple[float, float] | None:
        """returns the recommended velocity solution"""
        if not collidePointRect(rect, (self.posX, self.posY), inclusive=False):
            # print(no collide!")
            return

        rectBottom = rect[0][1] + rect[1][1]
        rectTop = rect[0][1]
        rectLeft = rect[0][0]
        rectRight = rect[0][0] + rect[1][0]

        selfTop = selfBottom = self.posY
        selfLeft = selfRight = self.posX

        bottomSol = (
            ((rectBottom) - selfTop) / (numpy.tan(self.wheelTheta)),
            rectBottom - selfTop,
            (0.0, 0.0),
        )

        topSol = (
            (rectTop - selfBottom) / (numpy.tan(self.wheelTheta)),
            rectTop - selfBottom,
            (0.0, 0.0),
        )

        leftSol = (
            (rectLeft - selfRight),
            (rectLeft - selfRight) * (numpy.tan(self.wheelTheta)),
            (0.0, 0.0),
        )

        rightSol = (
            (rectRight - selfLeft),
            (rectRight - selfLeft) * (numpy.tan(self.wheelTheta)),
            (0.0, 0.0),
        )

        solutions = [bottomSol, topSol, leftSol, rightSol]
        #
        # print(f"leftSol: {leftSol}")
        #
        # # solutions must move the robot in the opposite direction of velocity
        filteredSolutions = []
        for sol in solutions:
            vx = self.v * math.cos(self.wheelTheta)
            vy = self.v * math.sin(self.wheelTheta)
            if numpy.sign(vx) == -numpy.sign(sol[0]) and numpy.sign(vy) == -numpy.sign(
                sol[1]
            ) and collidePointRect(rect, numpy.add((self.posX, self.posY), sol[0:2]).tolist()):
                filteredSolutions.append(sol)
        #
        # # print(f"filteredSolutions: {filteredSolutions}")
        #
        # validSolutions = []
        # for solution in filteredSolutions:
        #     resultingRect = numpy.add(
        #         self.toRect(), ((solution[0], solution[1]), (0, 0))
        #     )
        #     resultingRectCenter = (
        #         resultingRect[0][0] + resultingRect[1][0] / 2.0,
        #         resultingRect[0][1] + resultingRect[1][1] / 2.0,
        #     )
        #
        #     rectCenter = (rect[0][0] + rect[1][0] / 2.0, rect[0][1] + rect[1][1] / 2.0)
        #
        #     distance = numpy.abs(numpy.subtract(resultingRectCenter, rectCenter))
        #
        #     # print(f"distance: {distance}")
        #     # print(f"self.length: {self.length}")
        #
        #     if (
        #         round(distance[0], 8) == round((self.length + rect[1][0]) / 2.0, 8)
        #         and round(distance[1], 8) <= round((self.length + rect[1][1]) / 2.0, 3)
        #     ) or (
        #         round(distance[1], 8) == round((self.length + rect[1][1]) / 2.0, 8)
        #         and round(distance[0], 8) <= round((self.length + rect[1][0]) / 2.0, 3)
        #     ):
        #         print(f"solution: {solution}")
        #         print(f"position: {self.toRect()}")
        #         print(f"wheelTheta: {math.degrees(self.wheelTheta)}")
        #         print(f"resultingRect: {resultingRect}")
        #         print(f"rect: {rect}")
        #         validSolutions.append(solution)
        #
        # if len(validSolutions) <= 0:
        #     print("Found no solutions for this collision!")
        #     return
        # elif len(validSolutions) > 1:
        #     print("Found multiple valid solutions for this collision!")
        # #
        # # apply the first (and hopefully only) solution
        # print(f"validSolution: {validSolutions[0]}")
        self.posX += filteredSolutions[0][0]
        self.posY += filteredSolutions[0][1]
        #
        self.v = 0.0
        #
        # # return recommended velocity multiplier
        # return validSolutions[0][2]

        return filteredSolutions[0][2]

    def _physics(self, grid, fieldWidth: float, fieldHeight: float):
        nRows = len(grid)
        nCols = len(grid[0])

        topLeftCol = numpy.clip(
            int((self.posX - self.length / 2.0) * (nCols / fieldWidth)), 0, nCols - 1
        )
        topLeftRow = numpy.clip(
            int((self.posY - self.length / 2.0) * (nRows / fieldHeight)), 0, nRows - 1
        )

        bottomRightCol = numpy.clip(
            int((self.posX + self.length / 2.0) * (nCols / fieldWidth)), 0, nCols - 1
        )
        bottomRightRow = numpy.clip(
            int((self.posY + self.length / 2.0) * (nRows / fieldHeight)), 0, nRows - 1
        )

        velMult = [1, 1]
        for col in numpy.arange(topLeftCol, bottomRightCol + 1):
            for row in numpy.arange(topLeftRow, bottomRightRow + 1):
                if int(grid[row][col]) == 1:
                    mult = self._resolveCollision(
                        (
                            (col * (fieldWidth / nCols), row * (fieldHeight / nRows)),
                            (fieldWidth / nCols, fieldHeight / nRows),
                        )
                    )

                    # print(f"dim: {(fieldWidth / nCols, fieldHeight / nRows)}")

                    # print(
                    #     f"rect: {((col * (fieldWidth / nCols), row * (fieldHeight / nRows)), (fieldWidth / nCols, fieldHeight / nRows),)}"
                    # )

                    if mult == None:
                        continue

                    velMult = numpy.multiply(velMult, mult)

        # vx = self.v * numpy.cos(self.wheelTheta) * velMult[0]
        # vy = self.v * numpy.sin(self.wheelTheta) * velMult[1]
        # # #
        # self.v = numpy.sqrt(vx ** 2 + vy ** 2)

    def step(self, F: float, theta: float, grid, fieldWidth: float, fieldHeight: float):

        self.v = self.v * numpy.cos(numpy.subtract(theta, self.wheelTheta))
        self.wheelTheta = theta

        time = 0
        while time < self.deltaT:
            time += self._forceSim(F, self.deltaT - time)
            self._physics(grid, fieldWidth, fieldHeight)
            # print(f"fielddim: {fieldWidth}, {fieldHeight}")

        # Ff = -numpy.sign(self.v) * self.muK * 9.81 * self.mass # very simplistic model for friction, might need to change, IDK
        # Fm = F
        #
        # D = 0.0
        # zeroT = numpy.divide(numpy.multiply(-self.v, self.mass), numpy.add(Ff, Fm))
        #
        # if (numpy.add(Ff, Fm)) == 0.0:
        #     print(f"zeroT is {zeroT}")
        #
        # if zeroT > 0.0 and zeroT < self.deltaT:
        #     a = numpy.divide(numpy.add(Fm, Ff), self.mass)
        #     D += numpy.multiply(self.v, zeroT) + 0.5 * numpy.multiply(a, numpy.power(zeroT, 2))
        #     # self.v = self.v + a * (self.deltaT - zeroT)
        #     self.v = 0.0
        #
        #     print(f"zeroT is {zeroT}")
        #     print(f"v is {self.v}")
        #
        #     Ff = self.muK * 9.81 * self.mass # very simplistic model for friction, might need to change, IDK
        #     if (numpy.abs(Fm) - Ff) > 0:
        #         a = numpy.sign(Fm) * ((numpy.abs(Fm) - Ff) / self.mass)
        #         D += self.v * zeroT + 0.5 * a * ((self.deltaT - zeroT) ** 2)
        #         self.v = self.v + a * (self.deltaT - zeroT)
        # else:
        #     a = ((Fm + Ff)/ self.mass)
        #     D += self.v * self.deltaT + 0.5 * a * (self.deltaT ** 2)
        #     self.v = a * self.deltaT + self.v
        #
        # self.posX += D * math.cos(self.wheelTheta)
        # self.posY += D * math.sin(self.wheelTheta)
