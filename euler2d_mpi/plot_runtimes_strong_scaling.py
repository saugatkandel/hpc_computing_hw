#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



nprocs = np.genfromtxt('strong_scaling_np.dat')



runtimes = np.genfromtxt('runtimes_strong_scaling.dat', usecols=[1])



plt.figure(figsize=[5,5])
plt.plot(nprocs, runtimes[0] / (runtimes * nprocs),  marker='o')
plt.xticks(nprocs)
plt.xlabel('nprocs')
plt.ylabel('Strong scaling efficiency')
plt.title('For strong scaling, efficiency should be const at 1')
plt.tight_layout()
plt.savefig('runtimes_strong_scaling.png')

