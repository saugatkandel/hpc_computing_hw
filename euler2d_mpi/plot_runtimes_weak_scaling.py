#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



params = np.genfromtxt('weak_scaling_params.dat')
nprocs, n = params.T



runtimes = np.genfromtxt('runtimes_weak_scaling.dat', usecols=[1])



plt.figure(figsize=[5,5])
plt.plot(nprocs, runtimes[0] / runtimes,  marker='o')
plt.xticks(nprocs)
plt.xlabel('nprocs')
plt.ylabel('Weak scaling efficiency')
plt.title('For weakscaling, efficiency should be const at 1')
plt.tight_layout()
plt.savefig('runtimes_weak_scaling.png')
#plt.show()

