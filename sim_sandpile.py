import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time

from util import Directions
from sandpile import Sandpile

N_grid = 20 #number of cells per side

MAXIMUM_GRAINS = 4
N_runs = 20*20*4#10000


X_POS_INIT = N_grid//2
Y_POS_INIT = N_grid//2
sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, MAX_STEPS=N_runs)



for i in range(N_runs):
    print(i)
    sandpile.step()

#print final grid
sandpile.print_grid()

# get avalance sizes
avalanche_sizes = np.array(sandpile.avalanche_sizes)

#plot histogram and loglog
bins = 10
hist_vals, x_recon = np.histogram(avalanche_sizes, bins=bins, density=True)

fig, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(avalanche_sizes,bins=bins, density=True)
axs_hist[1].loglog(x_recon[:-1],hist_vals,color='r',marker='o')

#manually log avalanche sizes and then plot histogram
log_avalanche_sizes = np.log10(avalanche_sizes)

bins = 10
hist_vals, x_recon = np.histogram(log_avalanche_sizes, bins=bins, density=True)

fig, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(log_avalanche_sizes,bins=bins, density=True)
axs_hist[1].plot(np.log10(x_recon[:-1]),hist_vals,color='r',marker='o')
plt.show()


