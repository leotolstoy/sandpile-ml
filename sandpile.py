import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
from time import time



N_grid = 20 #number of cells per side
grid = np.zeros((N_grid, N_grid))
MAXIMUM_GRAINS = 4
N_runs = 10
dt = 1./30 # 30 fps
frames = N_runs
I = 0
fig = plt.figure()
axs = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, N_grid), ylim=(0, N_grid))
axs.grid()

def step():
    # drop a grain on a uniformly selected location
    # select a random location
    x_coord_grain = random.randint(0, N_grid-1)
    y_coord_grain = random.randint(0, N_grid-1)

    # print(x_coord_grain, y_coord_grain)

    #increment the count at that location
    grid[x_coord_grain, y_coord_grain] += 1
    # return grid

def animate(i):
    print(i)
    step()
    img = axs.imshow(grid)
    global I
    I += 1
    # print(I)
    return img, 

# for i in range(N_runs):
#     grid = step(i, grid)
#     animate(grid)

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds


anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False)


# img = axs.imshow(grid)
plt.show()


