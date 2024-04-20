from sim import Sim

initialPos = [0.0, 0.0]
timeStep = 50 * 10e-3 # in seconds
mass = 45.3592 # kg
length = 0.4572 # m

sim = Sim(posX=initialPos[0], posY=initialPos[1], deltaT=timeStep, mass=mass, muK=0.0, length=length)
