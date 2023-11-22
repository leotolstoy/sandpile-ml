import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time



N_grid = 10 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

grid = np.zeros((N_grid, N_grid))
MAXIMUM_GRAINS = 4
N_runs = 100
dt = 1./30 # 30 fps
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

cmap = plt.cm.viridis
bounds = np.arange(0, MAXIMUM_GRAINS+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

class Sandpile():

    def __init__(self,N_grid=10, MAXIMUM_GRAINS=4):
        self.grid = np.zeros((N_grid, N_grid))
        self.N_grid = N_grid
        self.MAXIMUM_GRAINS = MAXIMUM_GRAINS

    def step(self,):
        self.drop_sandgrain()

    def drop_sandgrain(self,):
        # drop a grain on a uniformly selected location
        # select a random location
        x_coord_grain = random.randint(0, self.N_grid-1)
        y_coord_grain = random.randint(0, self.N_grid-1)

        # print(x_coord_grain, y_coord_grain)

        #increment the count at that location
        self.grid[x_coord_grain, y_coord_grain] += 1
        print(self.grid)


sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS)
    

def init():
    """initialize animation"""
    img = axs.imshow(sandpile.grid,cmap=cmap,norm=norm, origin="lower")
    plt.colorbar(img)
    

    return img,


def animate(i):
    print(i)
    sandpile.step()
    img = axs.imshow(sandpile.grid)
    global I
    I += 1
    # print(I)
    return img, 

# for i in range(N_runs):
#     grid = step(i, grid)
#     animate(grid)

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds


anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)


# img = axs.imshow(grid)
plt.show()


