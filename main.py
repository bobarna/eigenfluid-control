import math
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

# type shorthands
vec2i = ti.types.vector(2, ti.i32)
vec3i = ti.types.vector(3, ti.i32)
vec2f = ti.types.vector(2, ti.f32)
vec3f = ti.types.vector(3, ti.f32)

# We evaluate on a PIxPI domain size, but sample it more dense
original_domain_size = (math.pi, math.pi)
sampling_size = (32, 32)
dim = 2
# The sampled, discrete current velocity field
vel_field = ti.Vector.field(n=dim, dtype=ti.f32, shape=sampling_size)

# Number of basis fields to be used
N = 16
# math.isqrt(x) ~ math.floor(math.sqrt(x))
N_sqrt = math.isqrt(N)
# Velocity basis fields
basis = ti.types.struct(
    i = ti.i32, # index of the basis in the list of all the bases
    w = ti.f32, # omega, the coefficient for this basis
    k = vec2i, # k=(k_1, k_2), the "vector wave number"
    eig = ti.f32, # eigenvalue
    eig_inv = ti.f32, # 1/eig
    eig_inv_root = ti.f32, # 1/sqrt(eig)
    # stored discrete basis field 
    # TODO does it make sense to store it in a discretized form?
    # field = ti.types.matrix(sampling_size[0], sampling_size[1], ti.f32),
)
basis_fields = basis.field(shape=(N))

# Assume N is a perfect square, and use all basis fields
# with eigenvalues (k1,k2) up to (sqrt(N), sqrt(N))
def init_basis_fields():
    i = 0
    for k_1 in range(1, N_sqrt+1):
        for k_2 in range(1, N_sqrt+1):
            basis_fields[i].w = 0
            basis_fields[i].k = [k_1, k_2]
            basis_fields[i].eig = k_1**2 + k_2**2
            basis_fields[i].eig_inv = 1 / basis_fields[i].eig
            basis_fields[i].eig_inv_root = ti.rsqrt(basis_fields[i].eig)
            basis_fields[i].i = i
            i = i+1

def print_basis_fields():
    for i in range(N):
        b = basis_fields[i]
        print(f"i={b.i}, w={b.w}; ({b.k[0]},{b.k[1]}), eig={b.eig}")

# The used Laplacian eigenfunction.
# k = (k_1, k_2): vector wave number (eigen-value = (k_1**2 + k_2**2))
# This closed-form expression is for a PIxPI square domain.
@ti.func
def phi_2d_rect(k=vec2i, p=vec2f) -> vec2f:
    k_1, k_2, x, y = k[0], k[1], p[0], p[1]
    fact = -1 / (k_1**2 + k_2**2)
    k_1x, k_2y = k_1*x, k_2*y
    x = fact * (-k_2) * ti.sin(k_1x) * ti.cos(k_2y)
    y = fact * k_1 * ti.cos(k_1x) * ti.sin(k_2y)
    return vec2f(x, y)

# Laplacian eigenfunctions on a PIxPI square domain.
# - k = (k_1, k_2) are integers ("vector wave number")
# - p = (p_1, p_2, ...) is the point to be sampled in R^(dim)
@ti.kernel
def get_phi(k: vec2i, p: vec2f) -> vec2f:
    return phi_2d_rect(k, p)

# Fill the velocity field with values by combining the basis fields,
# meaning linearly combining them via the w coefficients
@ti.kernel
def calculate_superposition():
    for x, y in vel_field:
        # transform to [0..PI] range
        p = vec2f(x, y) / sampling_size * original_domain_size
        # sum up all base functions
        for i in range(N):
            b = ti.static(basis_fields)
            vel_field[x, y] += b[i].w * phi_2d_rect(b[i].k, p)

def plot_vel_field():
    np.set_printoptions(precision=2)
    # fig = plt.figure(figsize=(8,8), dpi=600)
    # print("current eigen values:", k1, k2)
    sample_points = np.linspace(0, original_domain_size[0], sampling_size[0])
    X, Y = np.meshgrid(sample_points, sample_points, indexing='ij' )
    U = vel_field.to_numpy()[:, :, 0]
    V = vel_field.to_numpy()[:, :, 1]
    plt.quiver(X, Y, U, V)
    plt.title(f"N = {N}, w = {str(basis_fields.w.to_numpy())}")
    # plt.axis('scaled')
    plt.show()
    # fig.savefig(f"test_render/test_{k1}_{k2}.png")
    # plt.close(fig)


# gui = ti.GUI("laplacian eigenfunctions test", sampling_size)
# while gui.running:
#     gui.set_image(vel_field)
#     gui.show()

def main():
    init_basis_fields()
    for x in range(N):
        for i in range(N):
            basis_fields[i].w = 1 if x == i else 0 #np.random.rand()
        # print_basis_fields()
        # reset vel_field
        vel_field.fill(0)
        calculate_superposition()
        plot_vel_field()

if __name__ == '__main__':
    main()
