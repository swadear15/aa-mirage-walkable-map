"""Convert Mirage game coordinates (x_pos, y_pos) to pixel coordinates on de_mirage_radar.png.

Valve's radar for de_mirage is a 1024 x 1024 image with these calibration values:

    pos_x = -3230   game x at the left edge of the image
    pos_y =  1713   game y at the top edge of the image
    scale =  5.0    game units per pixel

Pixel x grows to the right like game x. Pixel y grows downward while game y grows
upward, so y is flipped. Game z (height) is not used.

    px = (x_pos - pos_x) / scale
    py = (pos_y - y_pos) / scale

Usage:
    python game_to_image.py -1776 -1800
    from game_to_image import game_to_image; game_to_image(-1776, -1800)
"""

import sys

POS_X = -3230.0
POS_Y = 1713.0
SCALE = 5.0


def game_to_image(x_pos, y_pos):
    """Game (x, y) -> (px, py) on the 1024x1024 radar image."""
    return (x_pos - POS_X) / SCALE, (POS_Y - y_pos) / SCALE


def image_to_game(px, py):
    """Pixel (px, py) on the 1024x1024 radar image -> game (x, y)."""
    return px * SCALE + POS_X, POS_Y - py * SCALE


if __name__ == "__main__":
    x, y = float(sys.argv[1]), float(sys.argv[2])
    px, py = game_to_image(x, y)
    print(f"game ({x}, {y}) -> image ({px:.1f}, {py:.1f})")
