#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



data = np.fromfile('prob_parallel.out', dtype='float64')



plt.figure()
plt.plot(data)
plt.ylabel('probability')
plt.xlabel('number of steps (x10)')
plt.tight_layout()
plt.savefig('probabilities.png')
plt.show()
