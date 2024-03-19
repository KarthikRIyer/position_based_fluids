import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, cpu_max_num_threads=1)
# ti.init(arch=ti.gpu)

NUM_PARTICLES_ROW = 50
NUM_PARTICLES_COL = 30
NUM_PARTICLES = NUM_PARTICLES_ROW * NUM_PARTICLES_COL

# GUI
WIDTH = 600
HEIGHT = 600
BACKGROUND_COLOUR = 0xf0f0f0
PARTICLE_COLOUR = 0x328ac1
PARTICLE_RADIUS = 4

# parameters
SUBSTEPS = 2
SOLVE_ITERS = 10
MAX_PARTICLES_IN_A_GRID = 64
MAX_NEIGHBOURS = 64
KERNEL_SIZE = 15     # h in Poly6 kernel and Spiky kernel
KERNEL_SIZE_SQR = KERNEL_SIZE * KERNEL_SIZE
POLY6_CONST = 315 / 64 / np.pi / KERNEL_SIZE ** 9
SPIKY_GRAD_CONST = -45 / np.pi / KERNEL_SIZE ** 6
GRID_SIZE = KERNEL_SIZE
GRID_SHAPE = (WIDTH // GRID_SIZE + 1, HEIGHT // GRID_SIZE + 1)
PARTICLE_MASS = 1.0
RHO_0 = PARTICLE_MASS * (POLY6_CONST * (KERNEL_SIZE_SQR) ** 3) * 0.2
COLLISION_EPSILON = 1e-3
LAMBDA_EPSILON = 200
S_CORR_DELTA_Q = 0.3
S_CORR_K = 0.1
S_CORR_N = 4
S_CORR_CONST = 1 / (POLY6_CONST * (KERNEL_SIZE_SQR - KERNEL_SIZE_SQR * S_CORR_DELTA_Q * S_CORR_DELTA_Q) ** 3)
dt = 1 / 60 / SUBSTEPS
gravity = ti.Vector([0.0, -980])

# simulator variables
paused = False
attract = 0
mouse_pos = (0, 0)

# GPU variables
x = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
x_new = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
f = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
w = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
ni = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
niCrossG = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
gradWx = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
gradWy = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
gradWz = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
gradVx = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
gradVy = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
gradVz = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
lambda_ = ti.field(ti.f32, shape=NUM_PARTICLES)
rho_i = ti.field(ti.f32, shape=NUM_PARTICLES)
dx = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
grid = ti.field(ti.i32, shape=(GRID_SHAPE[0], GRID_SHAPE[1], MAX_PARTICLES_IN_A_GRID))
num_particles_in_grid = ti.field(ti.i32, shape=GRID_SHAPE)
neighbours = ti.field(ti.i32, shape=(NUM_PARTICLES, MAX_NEIGHBOURS))
num_neighbours = ti.field(ti.i32, shape=NUM_PARTICLES)
x_display = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)


@ti.kernel
def reset_particles():
    for i in range(NUM_PARTICLES_ROW):
        for j in range(NUM_PARTICLES_COL):
            # We add a random value so that they don't stack exact vertically
            # x[i * NUM_PARTICLES_COL + j][0] = 20 + j * KERNEL_SIZE * 0.2 + ti.random()
            # x[i * NUM_PARTICLES_COL + j][1] = 50 + i * KERNEL_SIZE * 0.2 + ti.random()
            x[i * NUM_PARTICLES_COL + j][0] = 20 + j * (WIDTH/NUM_PARTICLES_COL) * 0.7 + ti.random()
            x[i * NUM_PARTICLES_COL + j][1] = 50 + i * (HEIGHT/NUM_PARTICLES_ROW) * 0.7 + ti.random()
            v[i * NUM_PARTICLES_COL + j] = 0, 0
            w[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            gradWx[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            gradWy[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            gradWz[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            gradVx[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            gradVy[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            gradVz[i * NUM_PARTICLES_COL + j] = 0, 0, 0
            f[i * NUM_PARTICLES_COL + j] = 0, 0

    # for i in x:
    #     w[i] = ti.Vector([0.0, 0.0, 0.0])
    #     gradWx[i] = ti.Vector([0.0, 0.0, 0.0])
    #     gradWy[i] = ti.Vector([0.0, 0.0, 0.0])
    #     gradWz[i] = ti.Vector([0.0, 0.0, 0.0])
    #     gradVx[i] = ti.Vector([0.0, 0.0, 0.0])
    #     gradVy[i] = ti.Vector([0.0, 0.0, 0.0])
    #     gradVz[i] = ti.Vector([0.0, 0.0, 0.0])
    #     f[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def apply_external_forces(mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
    # alpha =
    for i in x_new:
        # v[i][1] = v[i][1] + dt * 980   # gravity
        v[i] = v[i] + dt * ((-0.9 * gravity) + (f[i]/PARTICLE_MASS))
        f[i] = ti.Vector([0.0, 0.0])
        # mouse interaction
        if attract:
            r = ti.Vector([mouse_x * WIDTH, mouse_y * HEIGHT]) - x[i]
            r_norm = r.norm()
            if r_norm > 15:
                v[i] += attract * dt * 5e6 * r / r_norm ** 3   # F = GMm/|r|^2 * (r/|r|)

        x_new[i] = x[i] + dt * v[i]
        gradWx[i] = ti.Vector([0, 0, 0])
        gradWy[i] = ti.Vector([0, 0, 0])
        gradWz[i] = ti.Vector([0, 0, 0])
        gradVx[i] = ti.Vector([0, 0, 0])
        gradVy[i] = ti.Vector([0, 0, 0])
        gradVz[i] = ti.Vector([0, 0, 0])

    box_collision()


@ti.kernel
def calculate_w():

    for x1 in x_new:
        ni[x1] = ti.Vector([0.0, 0.0, 0.0])
        # print(x1)
        for i in range(num_neighbours[x1]):
            x2 = neighbours[x1, i]
            r = x_new[x1] - x_new[x2]
            grad = spiky_grad_kernel(r)
            grad3 = ti.Vector([grad[0], grad[1], 0])
            gradWx[x1] += ((PARTICLE_MASS / rho_i[x2]) * (w[x2][0] * grad3))
            gradWy[x1] += ((PARTICLE_MASS / rho_i[x2]) * (w[x2][1] * grad3))
            gradWz[x1] += ((PARTICLE_MASS / rho_i[x2]) * (w[x2][2] * grad3))
            gradVx[x1] += ((PARTICLE_MASS / rho_i[x2]) * (v[x2][0] * grad3))
            gradVy[x1] += ((PARTICLE_MASS / rho_i[x2]) * (v[x2][1] * grad3))
            gradVz[x1] += ((PARTICLE_MASS / rho_i[x2]) * (0.0 * grad3))
            ni[x1] += ((PARTICLE_MASS / RHO_0) * (1.0 * grad3))
        velocity3 = ti.Vector([v[x1][0], v[x1][1], 0.0])
        vDotDelW = ti.Vector([velocity3.dot(gradWx[x1]), velocity3.dot(gradWy[x1]),
                              velocity3.dot(gradWz[x1])])
        # print(velocity3)
        # print(gradWx[x1])
        # print(vDotDelW)
        wDotDelV = ti.Vector(
            [w[x1].dot(gradVx[x1]), w[x1].dot(gradVy[x1]), w[x1].dot(gradVz[x1])])
        niCrossG[x1] = ni[x1].cross(ti.Vector([gravity[0], gravity[1], 0.0]))
        delWDelT = wDotDelV + (0.5 * niCrossG[x1]) - vDotDelW
        delW = (delWDelT * dt)
        w[x1] += delW
        # print(x1)
        # print(gradWx[x1])


@ti.kernel
def find_neighbours():
    """
    We look up neighbours in 3 steps:
    1. clear grid to particle table (set size to 0 for each cell)
    2. put each particle into a grid cell by its position
    3. for each particle, look for neighbour in its closest 9 grid cells, put into neighbour table

    Note on Taichi: the outer-most for loop in each kernel function is parallelized, therefore we need to use
    atomic add to increment shared table values
    """
    for i, j in num_particles_in_grid:
        num_particles_in_grid[i, j] = 0

    for i in x:
        grid_idx = int(x_new[i] / GRID_SIZE)
        old = ti.atomic_add(num_particles_in_grid[grid_idx], 1)
        if (old < MAX_PARTICLES_IN_A_GRID):
            grid[grid_idx[0], grid_idx[1], old] = i

    for x1 in x:
        neighbours_idx = 0
        grid_idx = int(x_new[x1] / GRID_SIZE)
        for grid_y in ti.static(range(-1, 2)):
            if 0 <= grid_idx[1] + grid_y < GRID_SHAPE[1]:
                for grid_x in ti.static(range(-1, 2)):
                    if 0 <= grid_idx[0] + grid_x < GRID_SHAPE[0]:
                        for i in range(num_particles_in_grid[grid_idx[0] + grid_x, grid_idx[1] + grid_y]):
                            x2 = grid[grid_idx[0] + grid_x, grid_idx[1] + grid_y, i]
                            if (x_new[x2] - x_new[x1]).norm_sqr() < KERNEL_SIZE_SQR and neighbours_idx < MAX_NEIGHBOURS and x2 != x1:
                                neighbours[x1, neighbours_idx] = x2
                                neighbours_idx += 1
        num_neighbours[x1] = neighbours_idx


@ti.func
def poly6_kernel(r_sqr):
    ret_val = 0.
    if r_sqr < KERNEL_SIZE_SQR:
        ret_val = POLY6_CONST * (KERNEL_SIZE_SQR - r_sqr) ** 3
    return ret_val


@ti.func
def spiky_grad_kernel(r):
    ret_val = ti.Vector([0., 0.])
    r_norm = r.norm()
    if 0 < r_norm < KERNEL_SIZE:
        ret_val = r / r_norm * SPIKY_GRAD_CONST * (KERNEL_SIZE - r_norm) ** 2
    return ret_val


@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)


@ti.kernel
def solve_iter():
    """
    We try to satisfy the density constraint here. We compute lambdas in the first loop, then compute dx
    (Delta p in the original paper) in the second part, finally update the position.
    This follows a non-linear Jacobi Iteration pattern. We run multiple solve_iter steps in a sub-step, and
    multiple sub-steps in a frame
    """
    for x1 in x_new:
        sum_grad_pk_C_sq = 0.
        # rho_i = poly6_kernel(0)
        rho_i[x1] = poly6_kernel(0)
        sum_grad_pi_C = ti.Vector([0., 0.])
        for i in range(num_neighbours[x1]):
            x2 = neighbours[x1, i]
            r = x_new[x1] - x_new[x2]
            grad = spiky_grad_kernel(r) / RHO_0
            sum_grad_pi_C += grad
            sum_grad_pk_C_sq += grad.norm_sqr()
            # rho_i += poly6_kernel(r.norm_sqr())
            rho_i[x1] += poly6_kernel(r.norm_sqr())
            # vorti city force
            fVort = (w[i].cross(ti.Vector([r[0], r[1], 0.0])) * poly6_kernel(r.norm_sqr()))
            f[x1] += ti.Vector([fVort[0], fVort[1]])
        # if (f[x1][0] == float('nan') or f[x1][1] == float('nan')):
        # print(f[x1])
        # drag force
        # f[x1] += 0.1 * v[x1] * (1.0 - (rho_i[x1] / RHO_0))

        # C_i = rho_i / RHO_0 - 1
        C_i = rho_i[x1] / RHO_0 - 1
        lambda_[x1] = -C_i / (sum_grad_pk_C_sq + sum_grad_pi_C.norm_sqr() + LAMBDA_EPSILON)



    # for x1 in x_new:
    #     ni[x1] = ti.Vector([0.0, 0.0, 0.0])
    #     for i in range(num_neighbours[x1]):
    #         x2 = neighbours[x1, i]
    #         r = x_new[x1] - x_new[x2]
    #         grad = spiky_grad_kernel(r)
    #         grad3 = ti.Vector([grad[0], grad[1], 0])
    #         gradWx[x1] += ((PARTICLE_MASS/rho_i[x2]) * (w[x2][0] * grad3))
    #         gradWy[x1] += ((PARTICLE_MASS/rho_i[x2]) * (w[x2][1] * grad3))
    #         gradWz[x1] += ((PARTICLE_MASS/rho_i[x2]) * (w[x2][2] * grad3))
    #         gradVx[x1] += ((PARTICLE_MASS / rho_i[x2]) * (v[x2][0] * grad3))
    #         gradVy[x1] += ((PARTICLE_MASS / rho_i[x2]) * (v[x2][1] * grad3))
    #         gradVz[x1] += ((PARTICLE_MASS / rho_i[x2]) * (0.0 * grad3))
    #         ni[x1] += ((PARTICLE_MASS / RHO_0) * (1.0 * grad3))
    #     velocity3 = ti.Vector([v[x1][0], v[x1][1], 0.0])
    #     vDotDelW = ti.Vector([velocity3.dot(gradWx[x1]), velocity3.dot(gradWy[x1]), velocity3.dot(gradWz[x1])])
    #     wDotDelV = ti.Vector([w[x1].dot(gradVx[x1]), w[x1].dot(gradVy[x1]), w[x1].dot(gradVz[x1])])
    #     niCrossG[x1] = ni[x1].cross(ti.Vector([gravity[0], gravity[1], 0.0]))
    #     delWDelT = wDotDelV + (1e-10 * niCrossG[x1]) - vDotDelW
    #     delW = (delWDelT * dt)
    #     w[x1] += delW
        # if (isnan(w[x1][2])):
        #         print(x1)
        # w[x1][0] = max(min(w[x1][0] + delW[0], 0.5), -0.5)
        # w[x1][1] = max(min(w[x1][1] + delW[1], 0.5), -0.5)
        # w[x1][2] = max(min(w[x1][2] + delW[2], 0.5), -0.5)
        # print(niCrossG[x1])
        # print(w[x1])
        # print(x1)
        # if (x1 == 1236):
            # print(w[x1])
            # print(ni[x1])

    for x1 in x_new:
        dx[x1] = ti.Vector([0., 0.])
        for i in range(num_neighbours[x1]):
            x2 = neighbours[x1, i]
            r = x_new[x1] - x_new[x2]
            # s_corr = -S_CORR_K * (poly6_kernel(r.norm_sqr()) * S_CORR_CONST) ** S_CORR_N
            s_corr = 0.0
            dx[x1] += (lambda_[x1] + lambda_[x2] + s_corr) * spiky_grad_kernel(r)

    for x1 in x:
        x_new[x1] = x_new[x1] + dx[x1] / RHO_0


@ti.func
def box_collision():
    """
    In position based dynamics, collision handling is just moving particles back into a valid position, its
    velocity will be implied by the position change
    We add a wall to left, right and bottom. We leave top open for more fun. Potential problem: particles outside of
    the screen does not belong to any grid cell, therefore they don't have liquid properties.
    """
    for i in x_new:
        if x_new[i][0] < PARTICLE_RADIUS:
            x_new[i][0] = PARTICLE_RADIUS + COLLISION_EPSILON * ti.random()
        if x_new[i][0] > WIDTH - PARTICLE_RADIUS:
            x_new[i][0] = WIDTH - PARTICLE_RADIUS - COLLISION_EPSILON * ti.random()
        if x_new[i][1] < PARTICLE_RADIUS:
            x_new[i][1] = PARTICLE_RADIUS + COLLISION_EPSILON * ti.random()
        if x_new[i][1] > HEIGHT - PARTICLE_RADIUS:
            x_new[i][1] = HEIGHT - PARTICLE_RADIUS + COLLISION_EPSILON * ti.random()


@ti.kernel
def update():
    for i in range(NUM_PARTICLES):
        v[i] = (x_new[i] - x[i]) / dt
        x[i] = x_new[i]
    box_collision()

    # The display uses coordinates from 0 to 1
    for i in range(NUM_PARTICLES):
        x_display[i][0] = x[i][0] / WIDTH
        x_display[i][1] = x[i][1] / HEIGHT


def simulate(mouse_pos, attract):
    for _ in range(SUBSTEPS):
        # find_neighbours()
        apply_external_forces(mouse_pos[0], mouse_pos[1], attract)
        find_neighbours()
        for _ in range(SOLVE_ITERS):
            solve_iter()
        calculate_w()
        update()


def render(gui):
    q = x_display.to_numpy()
    for i in range(NUM_PARTICLES):
        # col = int('%02x%02x%02x' % (min(int(abs(f[i][0])*256), 256), min(int(abs(f[i][1])*256 ), 256), 256), 16)
        r = int(abs(w[i][0]) * 1)
        g = int(abs(w[i][1]) * 1)
        b = int(abs(w[i][2]) * 1)
        # print(niCrossG[i])
        # b = int(0.0 * 255)
        if (i == 1236):
            r = 255
            g = 0
            b = 0
        col = int("{0:02x}{1:02x}{2:02x}".format(max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))), 16)
        # gui.circle(pos=q[i], color=PARTICLE_COLOUR, radius=PARTICLE_RADIUS)
        gui.circle(pos=q[i], color=col, radius=PARTICLE_RADIUS)
    gui.show()


if __name__ == '__main__':
    gui = ti.GUI('Position Based Fluid',
                 res=(WIDTH, HEIGHT), background_color=BACKGROUND_COLOUR)

    reset_particles()

    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                paused = not paused
            elif e.key == 'r':
                reset_particles()

        if gui.is_pressed(ti.GUI.RMB):
            mouse_pos = gui.get_cursor_pos()
            attract = 1
        elif gui.is_pressed(ti.GUI.LMB):
            mouse_pos = gui.get_cursor_pos()
            attract = -1
        else:
            attract = 0

        if not paused:
            simulate(mouse_pos, attract)
        # paused = True
        render(gui)
