import numpy as np
from matplotlib import pyplot as plt

# https://economicsfromthetopdown.com/2019/04/25/visualizing-power-law-distributions/
x = np.linspace(1e-1,1e1,1000)
y = x**-2

x_norm = x/np.sum(x)
y_norm = y/np.sum(y)

hist_vals, x_recon = np.histogram(y, density=True)

fig, axs = plt.subplots(3,1)
axs[0].plot(x,y)
axs[1].plot(x,y_norm)
# axs[2].hist(x_recon[:-1],y_recon,color='r')
axs[2].hist(y, density=True)

fig, axs1 = plt.subplots(3,1)
axs1[0].loglog(x,y)
axs1[1].loglog(x,y_norm)
axs1[2].loglog(x_recon[:-1],hist_vals,color='r',marker='o')
plt.show()