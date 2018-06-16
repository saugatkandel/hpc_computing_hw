#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



nvals = np.genfromtxt('runtimes_n_params.dat')
runtimes = np.genfromtxt('runtimes_n.dat', usecols=[1])

plt.figure(figsize=[5,5])
plt.plot(nvals, runtimes, marker='o')
plt.xlabel('N')
plt.ylabel('runtime(s)')
plt.title('Num proc 5')
plt.savefig('runtimes_n.png', bbox_layout='tight')

