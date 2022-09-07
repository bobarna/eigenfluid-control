import math
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

ti.init(arch=ti.vulkan)

# type shorthands
vec2i = ti.types.vector(2, ti.i32)
vec3i = ti.types.vector(3, ti.i32)
vec2f = ti.types.vector(2, ti.f32)
vec3f = ti.types.vector(3, ti.f32)

WINDOW_SIZE = (500, 500)
window = ti.ui.Window("Fluid Simulation", WINDOW_SIZE)
canvas = window.get_canvas()

# We evaluate on a PIxPI domain size, but sample it more dense
ORIGINAL_DOMAIN_SIZE = (float(math.pi), float(math.pi))
SAMPLING_SIZE = (30, 30)
DIM = 2
# The sampled, discrete current velocity field
vel_field = ti.Vector.field(n=DIM, dtype=ti.f32, shape=SAMPLING_SIZE)
# The points used for visualizing the current flow
# Each point has an (x,y) position
# color: initial color, that slowly fades away for previous time steps
visu_point_struct = ti.types.struct(
    pos=vec2f, # current position 0..1
    pos_orig=vec2f, # real size position (original_domain_size)
    color=vec3f
)
visu_points = visu_point_struct.field(shape=(10000))

# Number of basis fields to be used. Expected to be a perfect square
N = 10**2
# math.isqrt(x) ~ math.floor(math.sqrt(x))
N_sqrt = math.isqrt(N)
# Velocity basis fields
basis_struct = ti.types.struct(
    i = ti.i32, # index of the basis in the list of all the bases
    w = ti.f32, # omega, the coefficient for this basis
    k = vec2i, # k=(k_1, k_2), the "vector wave number"
    eig = ti.f32, # eigenvalue
    eig_inv = ti.f32,
    eig_inv_root = ti.f32, # 1/sqrt(-eig), note the sign!
    fact=ti.f32, #normalization factor for the basis function
    # stored discrete basis field 
    # TODO does it make sense to store it in a discretized form?
    # field = ti.types.matrix(SAMPLING_SIZE[0], SAMPLING_SIZE[1], ti.f32),
)
basis_fields = basis_struct.field(shape=(N))

# index_lookup[k_1, k_2] = index of base field in basis_fields[] array
index_lookup = ti.field(ti.i32, shape=(N_sqrt, N_sqrt))

# Assume N is a perfect square, and use all basis fields
# with eigenvalues (k1,k2) up to (sqrt(N), sqrt(N))
def init_basis_fields():
    i = 0
    for k_1 in range(1, N_sqrt+1):
        for k_2 in range(1, N_sqrt+1):
            basis_fields[i].w = 0
            basis_fields[i].k = [k_1, k_2] # k=(k_1, k_2) wave number
            basis_fields[i].eig = -(k_1**2 + k_2**2)
            # 1/eig
            basis_fields[i].eig_inv = 1/basis_fields[i].eig
            # inverse (reciprocal) square root of -eig (note the sign)
            # 1/sqrt(-eig)
            basis_fields[i].eig_inv_root = ti.rsqrt(-basis_fields[i].eig)
            # from scalable eigenfluids paper, with normalization
            basis_fields[i].fact = 2/math.pi*basis_fields[i].eig_inv_root
            # from original paper:
            # basis_fields[i].fact = 1 / basis_fields[i].eig_inv
            basis_fields[i].i = i
            index_lookup[k_1, k_2] = i
            i = i+1

# The used Laplacian eigenfunction.
# k = (k_1, k_2): vector wave number (eigen-value = (k_1**2 + k_2**2))
# This closed-form expression is for a PIxPI square domain.
@ti.func
def phi_2d_rect(k=vec2i, p=vec2f) -> vec2f:
    k_1, k_2, x, y = k[0], k[1], p[0], p[1]
    i = index_lookup[k_1, k_2]
    b = ti.static(basis_fields)
    k_1x, k_2y = k_1*x, k_2*y
    x = -b[i].fact * k_2 * ti.sin(k_1x) * ti.cos(k_2y)
    y = +b[i].fact * k_1 * ti.cos(k_1x) * ti.sin(k_2y)
    return vec2f(x, y)

# Laplacian eigenfunctions on a PIxPI square domain.
# - k = (k_1, k_2) are integers ("vector wave number")
# - p = (p_1, p_2, ...) is the point to be sampled in R^(dim)
@ti.kernel
def get_phi(k: vec2i, p: vec2f) -> vec2f:
    return phi_2d_rect(k, p)

# p \in [0..PI]
# Get the superposition of the current basis fields (velocity) at position p
# We are not sampling with the discretized version in order to achieve better
# accuracy, and better gradients
@ti.func
def get_velocity(p=vec2f) -> vec2f:
    vel = vec2f(0,0)
    for i in range(N):
        b = ti.static(basis_fields)
        vel += b[i].w * phi_2d_rect(b[i].k, p)
    return vel

# Fill the velocity field with values by combining the basis fields,
# meaning linearly combining them via the w coefficients
@ti.kernel
def calculate_superposition():
    for x, y in vel_field:
        # transform to [0..PI] range
        p = vec2f(x, y) / SAMPLING_SIZE * ORIGINAL_DOMAIN_SIZE
        # sum up all base functions
        for i in range(N):
            b = ti.static(basis_fields)
            vel_field[x, y] += b[i].w * phi_2d_rect(b[i].k, p)

# @ti.kernel
# def precalculate_advection():
#     for 

@ti.kernel
def init_visu_points():
    d = float(ORIGINAL_DOMAIN_SIZE[0])
    for i in visu_points:
        x, y = ti.random(), ti.random()
        x_orig, y_orig = x*d, y*d
        visu_points[i].pos = vec2f([x, y])
        visu_points[i].pos_orig = vec2f([x_orig, y_orig])

dt = 0.003
@ti.kernel
def step_visu_points():
    for i in visu_points:
        p = ti.static(visu_points.pos_orig)
        # first estimate of the slope
        k_1 = get_velocity(p[i])
        # simple euler step
        # p[i] += dt * k_1
        p_mid = p[i] + dt/2 * get_velocity(p[i])
        # predict tangent at midpoint
        # TODO at time t+dt/2
        k_2 = get_velocity(p_mid)
        # Midpoint == Runge Kutta 2nd order
        # p[i] += dt * k2
        # RK4
        # correct the estimate
        k_3 = get_velocity(p[i]+dt/2*k_2)
        # predict the slope with full step
        # TODO at time t+dt
        k_4 = get_velocity(p[i]+dt*k_3)
        # finally, perform step with weighted slopes
        p[i] += dt/6*(k_1 + 2*k_2 + 2*k_3 + k_4)


# Scale visu points into range 0..1
@ti.kernel
def calc_visu_display_pos():
    for i in visu_points:
        visu_points[i].pos = visu_points[i].pos_orig / ORIGINAL_DOMAIN_SIZE

# Plot the velocity field using PyPlot arrows
def plot_vel_field():
    np.set_printoptions(precision=2)
    # fig = plt.figure(figsize=(8,8), dpi=600)
    # print("current eigen values:", k1, k2)
    sample_points = np.linspace(0, ORIGINAL_DOMAIN_SIZE[0], SAMPLING_SIZE[0])
    X, Y = np.meshgrid(sample_points, sample_points, indexing='ij' )
    U = vel_field.to_numpy()[:, :, 0]
    V = vel_field.to_numpy()[:, :, 1]
    plt.quiver(X, Y, U, V)
    plt.title(f"N = {N}, w = {str(basis_fields.w.to_numpy())}")
    # plt.axis('scaled')
    plt.show(block=False)
    # fig.savefig(f"test_render/test_{k1}_{k2}.png")
    # plt.close(fig)

# Print the basis field setup to the console
def print_basis_fields():
    for i in range(N):
        b = basis_fields[i]
        print(f"i={b.i}, w={b.w}; ({b.k[0]},{b.k[1]}), eig={b.eig}")

curr_basis = 90
# TODO make this a taichi function to get gradients ?
def use_single_basis_field():
    vel_field.fill(0)
    basis_fields.w.fill(0)
    basis_fields[curr_basis].w = 1
    calculate_superposition()

def main():
    global curr_basis
    # fill up basis_fields details (e.g. k1,k2...)
    init_basis_fields()

    # use a single basis field for visualization
    use_single_basis_field()

    canvas.set_background_color((1.,1.,1.))

    init_visu_points()
    # print(visu_points)
    while window.running:
        # scale positions to range 0..1 for displaying
        calc_visu_display_pos()
        canvas.circles(visu_points.pos, color=(0.1,0.1,0.1), radius=0.003)
        for _ in range(40):
            step_visu_points()
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'j': curr_basis += 1
            if window.event.key == 'k': curr_basis -= 1
            if window.event.key == 'v': plot_vel_field()
            use_single_basis_field()
            print(f"curr_basis={curr_basis}")
        window.show()

    # Plot each basis field individually using PyPlot
    # for x in range(N):
    #     for i in range(N):
    #         basis_fields[i].w = 1 if x == i else 0 #np.random.rand()
        # print_basis_fields()
        # reset vel_field
        # vel_field.fill(0)
        # calculate_superposition()
        # plot_vel_field()

if __name__ == '__main__':
    main()
