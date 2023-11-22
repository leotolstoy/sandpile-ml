import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time


DO_ANIM = False
N_grid = 7 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

grid = np.zeros((N_grid, N_grid))
MAXIMUM_GRAINS = 4
N_runs = 10000
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

# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
# https://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
cmap = plt.cm.viridis
bounds = np.arange(0, MAXIMUM_GRAINS+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

class Sandpile():

    def __init__(self,N_grid=10, MAXIMUM_GRAINS=4):
        self.grid = np.zeros((N_grid, N_grid))
        self.N_grid = N_grid
        self.MAXIMUM_GRAINS = MAXIMUM_GRAINS
        self.avalanche_size = 0
        self.is_avalanching = False
        self.was_avalanching_before = False
        self.avalanche_sizes = []

    def step(self,):
        # determine if we should avalanche, based on if any of the grid values
        # are greater than the alloweable maximum grain number
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)

        if not self.is_avalanching:
            self.drop_sandgrain()
            
            if self.was_avalanching_before:
                self.avalanche_sizes.append(self.avalanche_size)
                self.was_avalanching_before = False
                self.avalanche_size = 0
                print(self.avalanche_sizes)


        else:
            self.avalanche()
            self.avalanche_size += 1
            self.was_avalanching_before = True
            



    def drop_sandgrain(self,):
        # drop a grain on a uniformly selected location
        # select a random location
        x_coord_grain = random.randint(0, self.N_grid-1)
        y_coord_grain = random.randint(0, self.N_grid-1)

        # print(x_coord_grain, y_coord_grain)

        #increment the count at that location
        self.grid[x_coord_grain, y_coord_grain] += 1
        # print(self.grid)

    def avalanche(self,):
        print('AVALANCHING')
        print(self.grid)
        # find indices where avalanching/unstable
        # returns a Nx2 array of xy coordinates where N is the number of indices over the maximum
        avalanche_idxs = np.argwhere(self.grid >= self.MAXIMUM_GRAINS)
        N_avalanche_ixs = avalanche_idxs.shape[0]

        # print(avalanche_idxs)

        #pick an unstable vertex at random
        rand_idx = np.random.randint(N_avalanche_ixs)
        # print(rand_idx)
        rand_unstable = avalanche_idxs[rand_idx,:]
        # print(rand_unstable)

        x_coord_unstable = rand_unstable[0]
        y_coord_unstable = rand_unstable[1]

        print(x_coord_unstable, y_coord_unstable)

        # topple the grid at this coordinate
        self.grid[x_coord_unstable, y_coord_unstable] -= self.MAXIMUM_GRAINS

        # increment neighboring vertex counts
        self.increment_neighbors(x_coord_unstable, y_coord_unstable)

        print('POST TOPPLE')
        print(self.grid)

        # input()

    def increment_neighbors(self, x_coord, y_coord):

        if (x_coord - 1) >= 0:
            self.grid[x_coord - 1, y_coord] += 1

        if (x_coord + 1) < self.N_grid:
            self.grid[x_coord + 1, y_coord] += 1
        
        if (y_coord - 1) >= 0:
            self.grid[x_coord, y_coord - 1] += 1

        if (y_coord + 1) < self.N_grid:
            self.grid[x_coord, y_coord + 1] += 1


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

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds

if DO_ANIM:
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)

else:
    for i in range(N_runs):
        print(i)
        sandpile.step()
# get avalance sizes
avalanche_sizes = np.array(sandpile.avalanche_sizes)

#plot histogram and loglog
hist_vals, x_recon = np.histogram(avalanche_sizes, density=True)

fig, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(avalanche_sizes, density=True)
axs_hist[1].loglog(x_recon[:-1],hist_vals,color='r',marker='o')
# img = axs.imshow(grid)
plt.show()


