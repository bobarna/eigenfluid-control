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
# sampling_size = (32, 32)
sampling_size = (64, 64)
dim = 2
velocity_sample_field = ti.Vector.field(n=dim, dtype=ti.f32, shape=sampling_size)

# The used Laplacian eigenfunction.
# k = (k_1, k_2): vector wave number (eigen-value = (k_1**2 + k_2**2))
# This closed-form expression is for a PIxPI square domain.

@ti.func
# def phi(k: vec2i, p: vec2f) -> vec2f:
def phi(k, p):
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
    return phi(k, p)

@ti.kernel
def fill_values(k: vec2i):
    for x, y in velocity_sample_field:
        p = vec2f(x, y) / sampling_size * original_domain_size
        value = phi(k, p)
        for k in ti.static(range(dim)):
            velocity_sample_field[x, y][k] = value[k]

for k1, k2 in [(a,b) for a in range(1, 10) for b in range(1,10)]:
    fig = plt.figure(figsize=(8,8), dpi=600)
    print("current eigen values:", k1, k2)
    fill_values(vec2i(k1,k2))
    sample_points = np.linspace(0, original_domain_size[0], sampling_size[0])
    X, Y = np.meshgrid(sample_points, sample_points, indexing='ij' )
    U = velocity_sample_field.to_numpy()[:, :, 0]
    V = velocity_sample_field.to_numpy()[:, :, 1]
    plt.quiver(X, Y, U, V)
    plt.title(f"k = ({k1}, {k2}), λ_k={-(k1**2+k2**2)}")
    plt.axis('scaled')
    # plt.show()
    fig.savefig(f"test_render/test_{k1}_{k2}.png")
    plt.close(fig)


# gui = ti.GUI("laplacian eigenfunctions test", sampling_size)
# while gui.running:
#     gui.set_image(velocity_sample_field)
#     gui.show()
