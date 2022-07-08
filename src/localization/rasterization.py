import math
from typing import List, Tuple

from data_types import Position


def rasterize_line_3d(pos1: Position, pos2: Position) -> List[Tuple[int, int, int]]:
    gx0idx = math.floor(pos1.x)
    gy0idx = math.floor(pos1.y)
    gz0idx = math.floor(pos1.z)

    gx1idx = math.floor(pos2.x)
    gy1idx = math.floor(pos2.y)
    gz1idx = math.floor(pos2.z)

    sx = 1 if gx1idx > gx0idx else (-1 if gx1idx < gx0idx else 0)
    sy = 1 if gy1idx > gy0idx else (-1 if gy1idx < gy0idx else 0)
    sz = 1 if gz1idx > gz0idx else (-1 if gz1idx < gz0idx else 0)

    gx = gx0idx
    gy = gy0idx
    gz = gz0idx

    gxp = gx0idx + (1 if gx1idx > gx0idx else 0)
    gyp = gy0idx + (1 if gy1idx > gy0idx else 0)
    gzp = gz0idx + (1 if gz1idx > gz0idx else 0)

    vx = 1 if pos2.x == pos1.x else pos2.x - pos1.x
    vy = 1 if pos2.y == pos1.y else pos2.y - pos1.y
    vz = 1 if pos2.z == pos1.z else pos2.z - pos1.z

    vxvy = vx * vy
    vxvz = vx * vz
    vyvz = vy * vz

    errx = (gxp - pos1.x) * vyvz
    erry = (gyp - pos1.y) * vxvz
    errz = (gzp - pos1.z) * vxvy

    derrx = sx * vyvz
    derry = sy * vxvz
    derrz = sz * vxvy

    points = []
    while True:
        points.append((gx, gy, gz))
        if gx == gx1idx and gy == gy1idx and gz == gz1idx:
            break
        xr = abs(errx)
        yr = abs(erry)
        zr = abs(errz)

        if sx != 0 and (sy == 0 or xr < yr) and (sz == 0 or xr < zr):
            gx += sx
            errx += derrx

        elif sy != 0 and (sz == 0 or yr < zr):
            gy += sy
            erry += derry
        elif sz != 0:
            gz += sz
            errz += derrz
    return points
