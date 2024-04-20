import math

class Sim:
    def __init__(self, posX: float, posY: float, deltaT: float, mass: float, muK: float, length: float):
        self.posX = posX
        self.posY = posY
        self.v = 0.0
        self.mass = mass
        self.wheelTheta = 0.0
        self.deltaT = deltaT
        self.muK = muK
        self.length = length

    def step(self, F: float, theta: float):

        self.v = self.v * math.cos(theta - self.wheelTheta)
        self.wheelTheta = theta

        Ff = -abs(self.v) / self.v * self.muK * 9.81 * self.mass # very simplistic model for friction, might need to change, IDK

        a = ((F + Ff)/ self.mass)
        D = self.v * self.deltaT + 0.5 * a * (self.deltaT ** 2)
        self.v = a * self.deltaT + self.v
        self.posX += D * math.cos(self.wheelTheta)
        self.posY += D * math.sin(self.wheelTheta)
