from imageio.core.util import np
from numpy.core.fromnumeric import shape
from sim import Sim
import pygame, sys, random
from pygame.locals import *
import numpy
import numpy as np
import json
import math
import csv
import imageio

from solver import GridInfo, Solver

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

taskInfo = pathInfo["task"]
taskStart = taskInfo["start"]
taskEnd = taskInfo["end"]

# TODO: max velocity

pygame.init()

# Background color (R, G, B)
BACKGROUND = (255, 255, 255)

# Game Setup
FPS = 60
fpsClock = pygame.time.Clock()
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 900

gridImage = imageio.v2.imread(fieldInfo["gridImage"])
print(gridImage.shape[0:2])

grid = numpy.zeros(shape=gridImage.shape[0:2], dtype=bool)
it = numpy.nditer(grid, flags=["multi_index"])
for x in it:
    (i, j) = it.multi_index
    grid[i][j] = gridImage[i][j][3]

solver = Solver(fieldWidth, fieldHeight, grid, taskStart, taskEnd)

WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Swerve Pathopt Solver GUI!")

# sim = Sim(
#     posX=initialPos[0],
#     posY=initialPos[1],
#     deltaT=timeStep,
#     mass=mass,
#     muK=muK,
#     length=length,
#     WINDOW=WINDOW,
# )

# cameraZoomBounds = (0.1, 10)
# cameraZoom = 1.0
# cameraZoomRate = 0.05
# cameraPosition = [0, 0]


def getFieldToWindowDimensions(
    fieldDimensions: tuple[float, float]
) -> tuple[float, float]:
    if (fieldDimensions[0] / fieldDimensions[1]) > (WINDOW_WIDTH / WINDOW_HEIGHT):
        return (WINDOW_WIDTH, WINDOW_WIDTH * (fieldDimensions[1] / fieldDimensions[0]))
    else:
        return numpy.round(
            (
                WINDOW_HEIGHT * (fieldDimensions[0] / fieldDimensions[1]),
                WINDOW_HEIGHT,
            )
        )


def drawArrow(length: float, pos: tuple[float, float], direction: np.ndarray):
    vec = np.array(direction[0:2])
    if np.linalg.norm(vec) == 0.0:
        vec = np.array((0.0, 0.0))
    else:
        vec = vec / np.linalg.norm(vec) * length
    startPos = irlToScreen(pos)[0:2].astype(int).tolist()
    endPos = (irlToScreen(np.add(pos, vec))).astype(int).tolist()
    pygame.draw.line(WINDOW, (0, 0, 0), startPos, endPos)
    pygame.draw.rect(WINDOW, (0, 255, 0), ((startPos[0] - 1, startPos[1] - 1), (2, 2)))


def toScreenCoord(coord):
    fieldScreenDims = getFieldToWindowDimensions((fieldWidth, fieldHeight))
    conv = numpy.divide(fieldScreenDims, (fieldWidth, fieldHeight))

    # return numpy.round(numpy.add(
    #     (
    #         numpy.multiply(
    #             numpy.subtract(numpy.multiply(coord, conv), cameraPosition), cameraZoom
    #         )
    #     ),
    #     (0 * WINDOW_WIDTH / 2, 0 * WINDOW_HEIGHT / 2),
    # ))
    return numpy.ceil(
        numpy.add(
            (numpy.multiply(numpy.subtract(numpy.multiply(coord, conv), 0.0), 1.0)),
            (0 * WINDOW_WIDTH / 2, 0 * WINDOW_HEIGHT / 2),
        )
    )


def irlToScreen(coord):
    fieldScreenDims = getFieldToWindowDimensions((fieldWidth, fieldHeight))
    conv = numpy.divide(fieldScreenDims, (fieldWidth, fieldHeight))

    return numpy.ceil(numpy.multiply(coord, conv))


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

    # segments: list[list[tuple[float, float]]] = [[toScreenCoord((sim.posX, sim.posY))]]

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
                # cameraZoom = numpy.clip(
                #     cameraZoom + cameraZoomRate * event.y,
                #     cameraZoomBounds[0],
                #     cameraZoomBounds[1],
                # )
                pass
        #     pygameImguiRenderer.process_event(event)
        #
        # pygameImguiRenderer.process_inputs()

        # Render (Draw) elements of the game
        WINDOW.fill(BACKGROUND)
        # GL.glClearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Handle mouse dragging
        if mouseDraggedPos != None:
            # print(f"Mouse dragged pos: {mouseDraggedPos}")
            # delta = numpy.subtract(pygame.mouse.get_pos(), mouseDraggedPos)
            # mouseDraggedPos = pygame.mouse.get_pos()
            # cameraPosition = numpy.subtract(cameraPosition, delta)
            # print(f"Camera Pos: {cameraPosition}")
            pass

        # drawGrid(450, 300)
        for i, row in enumerate(grid):
            for j, item in enumerate(row):
                if item != 0:
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

        for r in range(0, solver.dataGrid.shape[0], 2):
            for c in range(0, solver.dataGrid.shape[1], 2):
                gridItem: GridInfo = solver.dataGrid[r][c]

                drawArrow(0.3, gridItem.position, gridItem.optimalDirection)

        for neighborless in solver.neighborless:
            pygame.draw.rect(
                WINDOW, (0, 0, 255), (irlToScreen(neighborless.position), (3, 3))
            )

        # # Draw field rectangle
        # pygame.draw.rect(
        #     WINDOW,
        #     (100, 0, 0),
        #     (toScreenCoord((0, 0)), irlToScreen((fieldWidth, fieldHeight))),
        #     width=2,
        # )
        #
        # pygame.draw.rect(
        #     WINDOW,
        #     (0, 0, 0),
        #     (
        #         toScreenCoord(
        #             (sim.posX - sim.length / 2.0, sim.posY - sim.length / 2.0)
        #         ),
        #         irlToScreen((sim.length, sim.length)),
        #     ),
        # )

        # if stepIndex < len(path["steps"]):
        #
        #     if (
        #         stepIndex <= (len(path["steps"]) - 2)
        #         and tick == path["steps"][stepIndex + 1]["tick"]
        #     ):
        #         segments.append([toScreenCoord((sim.posX, sim.posY))])
        #         stepIndex += 1
        #
        #     # print((path["steps"][stepIndex]["force"], math.radians(path["steps"][stepIndex]["theta"])))
        #     sim.step(
        #         path["steps"][stepIndex]["force"],
        #         math.radians(path["steps"][stepIndex]["theta"]),
        #         grid,
        #         fieldWidth,
        #         fieldHeight
        #     )

        # segments[-1].append(toScreenCoord((sim.posX, sim.posY)))
        #
        # for i, segment in enumerate(segments):
        #     color = (255, 0, 0)
        #     if i % 3 == 1:
        #         color = (0, 255, 0)
        #     elif i % 3 == 2:
        #         color = (0, 0, 255)
        #     if len(segment) > 2:
        #         pygame.draw.lines(WINDOW, color, False, segment)
        #     elif len(segment) == 2:
        #         pygame.draw.line(WINDOW, color, segment[0], segment[1])

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
