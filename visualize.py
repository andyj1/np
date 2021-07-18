#!/usr/bin/env python3

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import LogNorm
# import matplotlib
# def f(x, y):
#     return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 40)

# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)       

# fig, ax = plt.subplots(111)
# plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
# plt.colorbar()
# plt.axis('auto');
# plt.show()

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
     

def f(x, y):
    # surrogate model
    z = (1 - x / 3. + x ** 5 + y ** 5) * np.exp(-x ** 2 - y ** 2)
    return z

dx, dy = 0.015, 0.05
x, y = np.mgrid[slice(-4, 4 + dx, dx),
                slice(-4, 4 + dy, dy)]
z = f(x,y)
z = z[:-1, :-1]
   
fig, ax = plt.subplots()
z_min, z_max = z.min(), z.max() #-np.abs(z).max(), np.abs(z).max()
c = ax.imshow(z, cmap ='RdGy', 
              vmin = z_min,
              vmax = z_max, extent =[x.min(),
                                     x.max(),
                                     y.min(),
                                     y.max()],
              interpolation ='nearest', 
              origin ='lower',
              label='density')

ax.scatter(-2,-2, s=50, marker='x', c='r', label='candidate')
fig.colorbar(c, ax = ax)
ax.set_title('example')
ax.legend(loc='best')
plt.show()