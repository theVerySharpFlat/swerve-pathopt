from sim import Sim
import pygame, sys, random
from pygame.locals import *
import numpy
import imgui
import imgui.integrations.pygame

import OpenGL.GL as GL

initialPos = [0.0, 0.0]
timeStep = 1.0 / 60.0  # in seconds
mass = 45.3592  # kg
length = 0.4572  # m

#           GR     torque  #motors    radius
maxForce = (6.12 * 1.068 * 4      ) / 0.1016

sim = Sim(
    posX=initialPos[0],
    posY=initialPos[1],
    deltaT=timeStep,
    mass=mass,
    muK=0.0,
    length=length,
)
pygame.init()

# Background color (R, G, B)
BACKGROUND = (1.0, 1.0, 1.0)

# Game Setup
FPS = 60
fpsClock = pygame.time.Clock()
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Swerve Pathopt GUI!")

cameraZoomBounds = (0.1, 10)
cameraZoom = 1.0
cameraZoomRate = 0.05
cameraPosition = [0, 0]


def toScreenCoord(coord):
    return numpy.add(
        numpy.multiply(numpy.subtract(coord, cameraPosition), cameraZoom),
        (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
    )


def irlToScreen(coord):
    return numpy.multiply(coord, 71)


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

    imgui.create_context()
    pygameImguiRenderer = imgui.integrations.pygame.PygameRenderer()
    imgui.get_io().display_size = WINDOW_WIDTH, WINDOW_HEIGHT
    imgui.get_io().fonts.add_font_default()


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
            pygameImguiRenderer.process_event(event)

        pygameImguiRenderer.process_inputs()

        # Render (Draw) elements of the game
        # WINDOW.fill(BACKGROUND)
        GL.glClearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Handle mouse dragging
        if mouseDraggedPos != None:
            print(f"Mouse dragged pos: {mouseDraggedPos}")
            delta = numpy.subtract(pygame.mouse.get_pos(), mouseDraggedPos)
            mouseDraggedPos = pygame.mouse.get_pos()
            cameraPosition = numpy.subtract(cameraPosition, delta)
            print(f"Camera Pos: {cameraPosition}")

        # drawGrid(450, 300)

        # Draw a rectangle!
        pygame.draw.rect(
            WINDOW,
            (0, 0, 0),
            (toScreenCoord((0, 0)), (20 * cameraZoom, 20 * cameraZoom)),
        )
        pygame.draw.rect(
            WINDOW,
            (255, 0, 0),
            ((-5 + WINDOW_WIDTH / 2, -5 + WINDOW_HEIGHT / 2), (10, 10)),
        )

        pygame.draw.rect(
            WINDOW,
            (0, 255, 0),
            (
                toScreenCoord(irlToScreen((sim.posX, sim.posY))),
                irlToScreen((sim.length, sim.length)),
            ),
        )
        sim.step(maxForce, 0.0)
        print((sim.posX, sim.posY))

        imgui.new_frame()

        # imgui.show_test_window()

        imgui.begin("Window")
        imgui.text("Hello, world!")
        imgui.end()
        # imgui.end_frame()

        imgui.render()

        # pygameImguiRenderer.render(imgui.get_draw_data())

        # Update the display!
        pygame.display.update()
        pygame.display.flip()

        # Update the clock limit framerate to FPS
        fpsClock.tick(FPS)


main()
