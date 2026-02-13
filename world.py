from panda3d.core import PNMImage, Texture
import math


def create_checker_texture(
    size: int = 128,
    cells: int = 8,
    color_a: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    color_b: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> Texture:
    image = PNMImage(size, size, 4)
    c0 = color_a
    c1 = color_b
    step = max(1, size // cells)
    for y in range(size):
        for x in range(size):
            ix = x // step
            iy = y // step
            c = c0 if (ix + iy) % 2 == 0 else c1
            image.setXelA(x, y, c[0], c[1], c[2], c[3])

    tex = Texture("checker-proc")
    tex.load(image)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    tex.setAnisotropicDegree(4)
    return tex


def create_shadow_texture(size: int = 128) -> Texture:
    image = PNMImage(size, size, 4)
    center = (size - 1) * 0.5
    inv_r = 1.0 / max(1.0, center)
    for y in range(size):
        for x in range(size):
            dx = (x - center) * inv_r
            dy = (y - center) * inv_r
            r = math.sqrt(dx * dx + dy * dy)
            if r >= 1.0:
                a = 0.0
            elif r <= 0.35:
                a = 0.72
            elif r <= 0.72:
                a = 0.46
            else:
                a = 0.2
            image.setXelA(x, y, 0.04, 0.04, 0.05, a)

    tex = Texture("ball-shadow-proc")
    tex.load(image)
    tex.setWrapU(Texture.WMClamp)
    tex.setWrapV(Texture.WMClamp)
    tex.setMinfilter(Texture.FTLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex
