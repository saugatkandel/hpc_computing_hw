#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import subprocess



array_sizes = []
runtimes = []
run_cmd = './cgl {N} 1.25 0.25 2000 500 test.out'
for i in range(1, 5):
    N = 64 * i
    cmd = run_cmd.format(N=N)
    output = (subprocess.check_output(cmd, shell=True)).decode()
    print(N, output)
    _, runtime = output.split()
    array_sizes.append(N)
    runtimes.append(float(runtime))

plt.figure()
plt.plot(array_sizes, runtimes)
plt.xlabel('Array dimension')
plt.ylabel('Time(s)')
plt.title('Run time for 2000 steps')
plt.tight_layout()
plt.savefig('runtimes.png')

subprocess.call(['rm test.out'], shell=True)

