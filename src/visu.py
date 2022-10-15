from phi.torch.flow import *
import matplotlib.pyplot as plt

# Helper for plotting w as a bar plot
def get_w_point_cloud(w):
    N = w.shape.get_size('k')
    x = math.range(instance(bars=N))
    # Scale horizontal axis
    # step = w.max / N
    # x *= w.max/N
    bar_heights = stack(w, instance('bars'))
    # to be called inside vis.plot(...)
    return PointCloud(Box(x=(x, x+1),
                          y=(0, bar_heights)),
                      bounds=Box(x=(0,N), # scaled horizontal axis
                                 y=(w.min,w.max))
                      )

def get_rescaled_vel(vel, VIS_DIM=10, DOMAIN=Box(x=math.PI, y=math.PI)):
    vis_grid = CenteredGrid(0, x=VIS_DIM, y=VIS_DIM, bounds=DOMAIN)
    return vel/vel.data.max * math.pi/VIS_DIM @ vis_grid

def get_visu_dict( vel=None,   vel_label='Velocity',
                   curl=None,  curl_label='Curl',
                   smoke=None, smoke_label='Smoke',
                   w=None,     w_label='w'):
    visu_dict = {}
    # Resampling velocity field on a coarser grid for visualization
    if vel is not None: visu_dict[vel_label] = get_rescaled_vel(vel)
    if curl is not None: visu_dict[curl_label] = field.curl(vel)
    if smoke is not None: visu_dict[smoke_label] = smoke
    if w is not None: visu_dict[w_label] = get_w_point_cloud(w)
    return visu_dict


