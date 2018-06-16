#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



data = np.fromfile('euler2d.out', 'float64')



N = 100
N2 = N**2
rho = data[:N2].reshape(N,N)
u = data[N2: 2 * N2].reshape(N,N)
v = data[2 * N2: 3 * N2].reshape(N, N)
p = data[3 * N2:].reshape(N,N)



plt.figure(figsize=[12,12])
plt.subplot(2,2,1)
plt.title('rho')
plt.contourf(rho)
plt.subplot(2,2,2)
plt.contourf(u)
plt.title('u')
plt.subplot(2,2,3)
plt.title('v')
plt.contourf(v)
plt.subplot(2,2,4)
plt.title('p')
plt.contourf(p)
plt.tight_layout()
plt.savefig('contour_plots.png')

