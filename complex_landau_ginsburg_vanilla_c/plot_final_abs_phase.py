#Author - Saugat Kandel
# coding: utf-8


# Plot the absolute value and phase after 10000 physical steps
# 128x128 array



import numpy as np
import subprocess
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt



try:
    data = np.fromfile('CGL.out', dtype='complex128')
except FileNotFoundError:
    subprocess.call(["make run_default"], shell=True)
    data = np.fromfile('CGL.out', dtype='complex128')



data_all_steps = data.reshape(11, 128, 128)
final_abs = np.abs(data_all_steps[-1])
final_angs = np.angle(data_all_steps[-1])



plt.figure(figsize=[8,4])
plt.subplot(1,2,1)
plt.contourf(final_abs)
plt.subplot(1,2,2)
plt.contourf(final_angs)
plt.tight_layout()
plt.savefig('final_abs_ang.png')
#plt.show()

