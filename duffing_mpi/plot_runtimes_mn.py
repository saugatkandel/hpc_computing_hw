#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



mn_params = np.genfromtxt('runtimes_mn_params.dat')



mvals = mn_params[:,0].reshape(5,5)
nvals = mn_params[:,1].reshape(5,5)



runtimes = np.genfromtxt('runtimes_mn.dat', usecols=[1])
runtimes = runtimes.reshape(5,5)



plt.figure(figsize=[12,5])
plt.subplot(1,2,1)
plt.plot(mvals.diagonal() * nvals.diagonal(), runtimes.diagonal(), marker='o')
plt.xlabel('M x N')
plt.ylabel('runtime(s)')
plt.subplot(1,2,2)
plt.contourf(mvals, nvals, runtimes)
plt.xlabel('M')
plt.ylabel('N')
cl = plt.colorbar()
cl.set_label('runtime(s)')
plt.suptitle('Num proc 5')
plt.savefig('runtimes_mn.png', bbox_layout='tight')

