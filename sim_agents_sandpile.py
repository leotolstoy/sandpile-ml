import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time

from util import Directions
from sandpile import Sandpile

DO_ANIM = False
N_grid = 10 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

MAXIMUM_GRAINS = 4
N_runs = 10000
frames = N_runs
I = 0
fig = plt.figure()


axs = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, N_grid-0.5), ylim=(0, N_grid-0.5))
# axs.grid()
# axs.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
axs.set_xticks(np.arange(-.5, N_grid, N_tick_step))
axs.set_yticks(np.arange(-.5, N_grid, N_tick_step))
# axs.set_xticks([])
# axs.set_yticks([])

axs.xaxis.set_tick_params(labelbottom=False)
axs.yaxis.set_tick_params(labelleft=False)

# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
# https://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
cmap = plt.cm.viridis
bounds = np.arange(0, MAXIMUM_GRAINS+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS)

def init():
    """initialize animation"""
    img = axs.imshow(sandpile.grid,cmap=cmap,norm=norm, origin="lower")
    plt.colorbar(img)
    
    return img,


def animate(i):
    # print(i)
    sandpile.step()
    img = axs.imshow(sandpile.grid)
    global I
    I += 1
    # print(I)
    return img, 

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds

if DO_ANIM:
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)

else:
    for i in range(N_runs):
        # print(i)
        sandpile.step()

# get avalance sizes
avalanche_sizes = np.array(sandpile.avalanche_sizes)

#plot histogram and loglog
bins = 10
hist_vals, x_recon = np.histogram(avalanche_sizes, bins=bins, density=True)

fig, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(avalanche_sizes,bins=bins, density=True)
axs_hist[1].loglog(x_recon[:-1],hist_vals,color='r',marker='o')

# img = axs.imshow(grid)
plt.show()


