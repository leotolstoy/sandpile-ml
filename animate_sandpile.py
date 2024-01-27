import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time, sleep

from util import Directions
from sandpile import Sandpile

N_grid = 10 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

MAXIMUM_GRAINS = 4
N_runs = 1000

I = 0
fig = plt.figure()

#these bounds show the sandpile
LIM_MIN = 1 - 0.5
LIM_MAX = N_grid-0.5

axs = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(LIM_MIN, LIM_MAX), ylim=(LIM_MIN, LIM_MAX))
# axs.grid()
axs.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

axs.set_xticks(np.arange(-.5, N_grid, N_tick_step))
axs.set_yticks(np.arange(-.5, N_grid, N_tick_step))

# axs.set_xticks(np.arange(-.5 + 1, N_grid+1, N_tick_step))
# axs.set_yticks(np.arange(-.5 + 1, N_grid+1, N_tick_step))
# axs.set_xticks([])
# axs.set_yticks([])

# axs.xaxis.set_tick_params(labelbottom=False)
# axs.yaxis.set_tick_params(labelleft=False)

# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
# https://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
# cmap = plt.cm.viridis
cmap = plt.cm.get_cmap('Blues')
bounds = np.arange(0, MAXIMUM_GRAINS+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

X_POS_INIT = N_grid//2
Y_POS_INIT = N_grid//2
sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, MAX_STEPS=N_runs, STORE_STATE_BUFFER=True)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)



# run the simulation
for i in range(N_runs):
    print(i)
    
    sandpile.step()
    # print(sandpile.grid_buffer[i])
    # sleep(2)

grid_buffer = sandpile.grid_buffer
print(len(grid_buffer))

# for i in range(len(grid_buffer)):
#     # print(i)
#     print(grid_buffer[i])
#     # input()
# input()

# loop through the grid buffer
frames = len(grid_buffer)
def init():
    """initialize animation"""
    img = axs.imshow(grid_buffer[0],cmap=cmap,norm=norm, origin="lower")
    # fig.colorbar(img)
    
    return img,


def animate(i):
    # print(i)
    img = axs.imshow(grid_buffer[i],cmap=cmap,norm=norm, origin="lower")
    # print(grid_buffer[i])
    # sleep(2)
    return img, 

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)

#print final grid
sandpile.print_grid()

# get avalance sizes
avalanche_sizes = np.array(sandpile.avalanche_sizes)

#plot histogram and loglog
bins = 10
hist_vals, x_recon = np.histogram(avalanche_sizes, bins=bins, density=True)

fig_hist, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(avalanche_sizes,bins=bins, density=True)
axs_hist[1].loglog(x_recon[:-1],hist_vals,color='r',marker='o')

# img = axs.imshow(grid)


#manually log avalanche sizes and then plot histogram
log_avalanche_sizes = np.log10(avalanche_sizes)

bins = 10
hist_vals, x_recon = np.histogram(log_avalanche_sizes, bins=bins, density=True)

fig_hist, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(log_avalanche_sizes,bins=bins, density=True)
axs_hist[1].plot(np.log10(x_recon[:-1]),hist_vals,color='r',marker='o')
plt.show()

