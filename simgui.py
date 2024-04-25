from sim import Sim
import pygame, sys, random
from pygame.locals import *
import numpy
import json
import math
import csv

pathInfo = None
with open("path.json", "r") as pathJSON:
    pathInfo = json.loads(pathJSON.read())

path = pathInfo["path"]
robotInfo = pathInfo["robot"]
fieldInfo = pathInfo["field"]

initialPos = path["start"]
timeStep = path["timeStep"]  # in seconds
mass = robotInfo["mass"]  # kg
length = robotInfo["sideLength"]  # m
maxForce = robotInfo["maxForce"]
muK = robotInfo["muK"]

fieldWidth = fieldInfo["width"]
fieldHeight = fieldInfo["height"]

# TODO: max velocity

pygame.init()

# Background color (R, G, B)
BACKGROUND = (255, 255, 255)

# Game Setup
FPS = 60
fpsClock = pygame.time.Clock()
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Swerve Pathopt GUI!")

sim = Sim(
    posX=initialPos[0],
    posY=initialPos[1],
    deltaT=timeStep,
    mass=mass,
    muK=muK,
    length=length,
    WINDOW=WINDOW,
)

cameraZoomBounds = (0.1, 10)
cameraZoom = 1.0
cameraZoomRate = 0.05
cameraPosition = [0, 0]


def getFieldToWindowDimensions(
    fieldDimensions: tuple[float, float]
) -> tuple[float, float]:
    if (fieldDimensions[0] / fieldDimensions[1]) > (WINDOW_WIDTH / WINDOW_HEIGHT):
        return (WINDOW_WIDTH, WINDOW_WIDTH * (fieldDimensions[1] / fieldDimensions[0]))
    else:
        return (
            WINDOW_HEIGHT * (fieldDimensions[0] / fieldDimensions[1]),
            WINDOW_HEIGHT,
        )


def toScreenCoord(coord):
    fieldScreenDims = getFieldToWindowDimensions((fieldWidth, fieldHeight))
    conv = numpy.divide(fieldScreenDims, (fieldWidth, fieldHeight))

    return numpy.round(numpy.add(
        (
            numpy.multiply(
                numpy.subtract(numpy.multiply(coord, conv), cameraPosition), cameraZoom
            )
        ),
        (0 * WINDOW_WIDTH / 2, 0 * WINDOW_HEIGHT / 2),
    ))


def irlToScreen(coord):
    fieldScreenDims = getFieldToWindowDimensions((fieldWidth, fieldHeight))
    conv = numpy.divide(fieldScreenDims, (fieldWidth, fieldHeight))

    return numpy.round(numpy.multiply(coord, conv))


def drawGrid(gapX: float, gapY: float):

    topLeft = numpy.divide(
        toScreenCoord(
            (
                cameraPosition[0]
                - WINDOW_WIDTH / 2.0 / cameraZoom
                - (cameraPosition[0] % gapX),  # / cameraZoom,
                cameraPosition[1]
                - WINDOW_HEIGHT / 2.0 / cameraZoom
                - (cameraPosition[1] % gapY),  # / cameraZoom,
            )
        ),
        cameraZoom**0,
    )
    bottomRight = numpy.divide(
        toScreenCoord(
            (
                cameraPosition[0] + WINDOW_WIDTH / 2.0 / cameraZoom,
                # - (cameraPosition[0] % gapX) / cameraZoom,
                cameraPosition[1] + WINDOW_HEIGHT / 2.0 / cameraZoom,
                # - (cameraPosition[1] % gapY) / cameraZoom,
            )
        ),
        cameraZoom**0,
    )

    gapX *= cameraZoom
    gapY *= cameraZoom

    xRange = (topLeft[0], bottomRight[0])
    yRange = (topLeft[1], bottomRight[1])

    cameraScreenCoords = toScreenCoord((0, 0))

    normalColor = (100, 100, 100)
    normalWidth = 1

    boldColor = (0, 0, 0)
    boldWidth = 1

    color = normalColor
    width = normalWidth
    for i in numpy.arange(xRange[0], xRange[1], gapX):
        if i == cameraScreenCoords[0]:
            color = boldColor
            width = boldWidth
        else:
            color = normalColor
            width = normalWidth

        pygame.draw.line(WINDOW, color, (i, 0), (i, WINDOW_HEIGHT), width)

    for i in numpy.arange(yRange[0], yRange[1], gapY):
        if i == cameraScreenCoords[1]:
            color = boldColor
            width = boldWidth
        else:
            color = normalColor
            width = normalWidth

        pygame.draw.line(WINDOW, color, (0, i), (WINDOW_WIDTH, i), width)

    # pygame.draw.line(
    #     WINDOW,
    #     (255, 0, 0),
    #     (cameraScreenCoords[0], 0),
    #     (cameraScreenCoords[0], WINDOW_HEIGHT),
    # )
    # pygame.draw.line(
    #     WINDOW,
    #     (255, 0, 0),
    #     (0, cameraScreenCoords[1]),
    #     (WINDOW_WIDTH, cameraScreenCoords[1]),
    # )


# The main function that controls the game
def main():
    global cameraPosition
    global cameraZoom
    mouseDraggedPos = None

    # imgui.create_context()
    # pygameImguiRenderer = imgui.integrations.pygame.PygameRenderer()
    # imgui.get_io().display_size = WINDOW_WIDTH, WINDOW_HEIGHT
    # imgui.get_io().fonts.add_font_default()

    segments: list[list[tuple[float, float]]] = [[toScreenCoord((sim.posX, sim.posY))]]

    grid = []
    with open(fieldInfo["gridCSV"]) as file:
        reader = csv.reader(file, delimiter=" ")

        for row in reader:
            grid.append(row)
    # print(grid)

    tick = 0
    stepIndex = 0

    # The main game loop
    while True:
        # Get inputs
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    mouseDraggedPos = pygame.mouse.get_pos()
            elif event.type == MOUSEBUTTONUP:
                if not pygame.mouse.get_pressed()[0]:
                    mouseDraggedPos = None
            elif event.type == pygame.MOUSEWHEEL:
                cameraZoom = numpy.clip(
                    cameraZoom + cameraZoomRate * event.y,
                    cameraZoomBounds[0],
                    cameraZoomBounds[1],
                )
        #     pygameImguiRenderer.process_event(event)
        #
        # pygameImguiRenderer.process_inputs()

        # Render (Draw) elements of the game
        WINDOW.fill(BACKGROUND)
        # GL.glClearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Handle mouse dragging
        if mouseDraggedPos != None:
            print(f"Mouse dragged pos: {mouseDraggedPos}")
            delta = numpy.subtract(pygame.mouse.get_pos(), mouseDraggedPos)
            mouseDraggedPos = pygame.mouse.get_pos()
            cameraPosition = numpy.subtract(cameraPosition, delta)
            print(f"Camera Pos: {cameraPosition}")

        # drawGrid(450, 300)
        for i, row in enumerate(grid):
            for j, item in enumerate(row):
                if int(item) != 0:
                    pygame.draw.rect(
                        WINDOW,
                        (255, 0, 0),
                        (
                            toScreenCoord(
                                (
                                    j * fieldWidth / len(row),
                                    i * fieldHeight / len(grid),
                                )
                            ),
                            irlToScreen(
                                (
                                    fieldWidth / len(row),
                                    fieldHeight / len(grid),
                                )
                            ),
                        ),
                    )

        # Draw field rectangle
        pygame.draw.rect(
            WINDOW,
            (100, 0, 0),
            (toScreenCoord((0, 0)), irlToScreen((fieldWidth, fieldHeight))),
            width=2,
        )

        pygame.draw.rect(
            WINDOW,
            (0, 0, 0),
            (
                toScreenCoord(
                    (sim.posX - sim.length / 2.0, sim.posY - sim.length / 2.0)
                ),
                irlToScreen((sim.length, sim.length)),
            ),
        )

        if stepIndex < len(path["steps"]):

            if (
                stepIndex <= (len(path["steps"]) - 2)
                and tick == path["steps"][stepIndex + 1]["tick"]
            ):
                segments.append([toScreenCoord((sim.posX, sim.posY))])
                stepIndex += 1

            # print((path["steps"][stepIndex]["force"], math.radians(path["steps"][stepIndex]["theta"])))
            sim.step(
                path["steps"][stepIndex]["force"],
                math.radians(path["steps"][stepIndex]["theta"]),
                grid,
                fieldWidth,
                fieldHeight
            )

        segments[-1].append(toScreenCoord((sim.posX, sim.posY)))

        for i, segment in enumerate(segments):
            color = (255, 0, 0)
            if i % 3 == 1:
                color = (0, 255, 0)
            elif i % 3 == 2:
                color = (0, 0, 255)
            if len(segment) > 2:
                pygame.draw.lines(WINDOW, color, False, segment)
            elif len(segment) == 2:
                pygame.draw.line(WINDOW, color, segment[0], segment[1])


        # imgui.new_frame()

        # imgui.show_test_window()

        # imgui.begin("Window")
        # imgui.text("Hello, world!")
        # imgui.end()
        # imgui.end_frame()

        # imgui.render()

        # pygameImguiRenderer.render(imgui.get_draw_data())

        # Update the display!
        pygame.display.update()
        pygame.display.flip()

        # Update the clock limit framerate to FPS
        tick += 1
        fpsClock.tick(FPS)


if __name__ == "__main__":
   main() 
