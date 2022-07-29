import numpy as np
import matplotlib.pyplot as plt
import math
import sys

DIM = 32

u = np.zeros((DIM,DIM,2))

file_name = "output"
if len(sys.argv) > 1:
    file_name = sys.argv[1]

with open(file_name) as f:
    for l in f:
        i, j = l.split('=')[0].split(",")
        i, j = int(i), int(j)
        x, y = l.split('=')[1].split(",")
        x, y = float(x), float(y)
        u[i,j,0] = x
        u[i,j,1] = y

# fig = plt.figure(dpi=500)
x, y = np.meshgrid(np.linspace(0, math.pi, DIM), np.linspace(0, math.pi, DIM),
                   indexing='ij')
plt.quiver(x, y, u[:,:,0], u[:,:,1])
# # plt.title(f"k = ({k1}, {k2}), λ_k={-(k1**2+k2**2)}")
# # fig.savefig(f"visu_{file_name}.png")
plt.show()
