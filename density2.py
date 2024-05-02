import math
from PIL import Image, ImageDraw

xo = 0.0
yo = 0.0
# rad = 0.0125
rad = 4
# KERNEL_SIZE = 0.15
KERNEL_SIZE = 15
KERNEL_SIZE_SQR = KERNEL_SIZE * KERNEL_SIZE
v = rad * math.sqrt(3.0)
# pmass = 0.25
pmass = 1.0
POLY6_CONST = 315 / 64 / math.pi / KERNEL_SIZE ** 9
rho_0 = pmass * (1.0 * POLY6_CONST * (KERNEL_SIZE_SQR) ** 3)

points = []


def poly6_kernel(r_sqr):
    ret_val = 0.0
    if r_sqr < KERNEL_SIZE_SQR:
        ret_val = POLY6_CONST * (KERNEL_SIZE_SQR - r_sqr) ** 3
    return ret_val


for i in range(0, 40):
    for j in range(0, 40):
        x = xo + 2.0*rad*i
        y = yo + 2.0*rad*j
        e = [x, y]
        points.append(e)

xo = xo + rad
yo = yo + rad


# for i in range(0, 40):
#     for j in range(0, 40):
#         x = xo + 2.0*rad*i
#         y = yo + 2.0*v*j
#         e = [x, y]
#         points.append(e)


xi = 20
yi = 20
px = 2.0*rad*xi
py = 2.0*rad*yi

maxx = -float("infinity")
maxy = -float("infinity")

density = 0.0
ncount = 0
for p in points:
    maxx = max(maxx, p[0])
    maxy = max(maxy, p[1])
    dist_sqr = ((p[0] - px)*(p[0] - px) + (p[1] - py)*(p[1] - py))
    if dist_sqr < KERNEL_SIZE_SQR:
        print('dist_sqr', dist_sqr)
        ncount += 1
        density += pmass * poly6_kernel(dist_sqr)

print('poly6(0): ', poly6_kernel(0.0))
print('ncount: ', ncount)
print('density max pack: ', density)
# print('rho_0:', rho_0)
print('density/poly6(0):', density/poly6_kernel(0.0))
print('maxx:', maxx)
print('maxy:', maxy)

scale = 5
image = Image.new(mode="RGBA", size = (400*scale, 400*scale))
image.paste((255, 255, 255), (0,0,400*scale,400*scale))
draw = ImageDraw.Draw(image)
for p in points:
    leftUpPoint = ((p[0] - rad)*scale, (p[1] - rad)*scale)
    rightDownPoint = ((p[0] + rad)*scale, (p[1] + rad)*scale)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, outline=(255,0,0,255))
    draw.point((p[0]*scale, p[1]*scale), fill='red')

leftUpPoint = ((px - KERNEL_SIZE)*scale, (py - KERNEL_SIZE)*scale)
rightDownPoint = ((px + KERNEL_SIZE)*scale, (py + KERNEL_SIZE)*scale)
draw.ellipse([leftUpPoint, rightDownPoint], outline="blue")
#
# leftUpPoint = ((px+2*rad - rad)*scale, (py - rad)*scale)
# rightDownPoint = ((px+2*rad + rad)*scale, (py + rad)*scale)
# draw.ellipse([leftUpPoint, rightDownPoint], outline="blue")
#
# leftUpPoint = ((px+2*rad - rad)*scale, (py+2*v - rad)*scale)
# rightDownPoint = ((px+2*rad + rad)*scale, (py+2*v + rad)*scale)
# draw.ellipse([leftUpPoint, rightDownPoint], outline="blue")
print(v)
image.save("test.png")