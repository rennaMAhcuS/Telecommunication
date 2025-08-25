from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from numpy import ndarray


def display_init() -> None:
    # PYGAME INIT
    pygame.init()
    screen = pygame.display.set_mode(flags=pygame.FULLSCREEN)
    screen.fill("red")
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
                break


def display_msg(frames: ndarray, ini_delay: int, delay: int,
                calibrate: bool = False) -> None:
    running = True
    screen = pygame.display.set_mode(flags=pygame.FULLSCREEN)

    # Show a green screen for 1 sec
    if not calibrate:
        screen.fill("green")
        pygame.display.flip()
        pygame.time.delay(1000 * ini_delay)

    for frame in frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        screen.fill("black")

        height, width = screen.get_height(), screen.get_width()
        rows, cols = frame.shape
        cell_size = min(width // cols, height // rows)
        x_offset = (width - cols * cell_size) // 2
        y_offset = (height - rows * cell_size) // 2
        for row in range(rows):
            for col in range(cols):
                color = "white" if frame[row, col] == 1 else "black"
                pygame.draw.rect(screen, color, (x_offset + col * cell_size,
                                                 y_offset + row * cell_size,
                                                 cell_size, cell_size))
        pygame.display.flip()
        pygame.time.delay(1000 * delay)

    if calibrate:
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
                    break

    # Go back to the red screen
    screen.fill("red")
    pygame.display.flip()


def display_end() -> None:
    # PYGAME QUIT
    pygame.quit()


if __name__ == "__main__":
    display_init()
    display_end()
