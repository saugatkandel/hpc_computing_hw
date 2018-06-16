#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import subprocess
import matplotlib.pyplot as plt



def singleRun(N, omega, maxiter):
    run_cmd = './source_sink -N {} -o {} -m {} -p'.format(N, omega, maxiter)
    output = (subprocess.check_output(run_cmd, shell=True)).decode()
    _, iteration, __, runtime = output.split()
    return int(iteration), float(runtime)

def calcBestOmega(N, maxiter, n_test_omegas):
    # Testing 50 different values of omega between 0.01 and 1.99

    print("Testing {} omega values between 0.01 and 1.99 for N = {}".format(n_test_omegas, N) 
          + " for a max number of iterations {}".format(maxiter)
          + " to find the best omega value...")

    
    omega_vals = np.linspace(0.01, 1.99, n_test_omegas)
    iterations = []
    runtimes = []
    for omega in omega_vals:
        iteration, runtime = singleRun(N, omega, maxiter)
        iterations.append(int(iteration))
        runtimes.append(float(runtime))

    min_iter_index = np.argmin(iterations)
    min_iter = np.min(iterations)
    min_omega = omega_vals[min_iter_index]
    min_runtime = runtimes[min_iter_index]

    return min_iter, min_omega, min_runtime
    
def calcBestOmegaWithinTolerance(N, maxiter, n_test_omegas):
    maxiter_this = maxiter
    best_iter, best_omega, best_runtime = calcBestOmega(N, maxiter_this, n_test_omegas)

    while best_iter >= maxiter_this:
        maxiter_this *= 2
        best_iter, best_omega, best_runtime = calcBestOmega(N, maxiter_this, n_test_omegas)

    print("for N = {}, the best omega value is {}".format(N, best_omega))
    print()
    print()
    return best_iter, best_omega, best_runtime




# For small even values of N, the operations complete in very small number of iterations
# which does not give any meaningful statistics
N_tests = np.arange(17, 50, 6)

# default maximum number of iterations
maxiter = 1000

# Test 50 omega values to find the best omega
# Ensure that the tolerance is always met. Increase maximum iteration limit if necessary.
best_iters = []
best_omegas = []
best_runtimes = []




for N in N_tests:
    best_iter, best_omega, best_runtime = calcBestOmegaWithinTolerance(N, maxiter, 50)
    best_iters.append(best_iter)
    best_omegas.append(best_omega)
    best_runtimes.append(best_runtime)



N_tests_2 = N_tests * 2
best_iters_2 = []
best_omegas_2 = []
best_runtimes_2 = []
for N in N_tests_2:
    best_iter, best_omega, best_runtime = calcBestOmegaWithinTolerance(N, maxiter, 50)
    best_iters_2.append(best_iter)
    best_omegas_2.append(best_omega)
    best_runtimes_2.append(best_runtime)



# Checking whether the omega value stays constant for N x (2N - 1) and 2N x (4N -1) grids
A1 = np.vstack([best_omegas, np.ones(np.size(best_omegas))]).T
out1 = np.linalg.lstsq(A1, best_omegas_2)
m1, c1 = out1[0]
resid1 = out1[1][0]

plt.figure()
plt.scatter(best_omegas, best_omegas_2)
plt.plot(best_omegas, np.array(best_omegas) * m1 + c1)
plt.xlabel('Best omegas for N x (2N - 1) grid')
plt.ylabel('Best omegas for 2N x (4N - 1) grid')
plt.title('Residual of linear fit {:3.2f} - omega does not stay the same'.format(resid1))
plt.tight_layout()
plt.savefig('omega_N_vs_2N.png')
plt.show()



# Testing whether the runtime per iteration scales with N

N_tests_combined = np.concatenate([N_tests, N_tests_2])
iterations = np.concatenate([best_iters, best_iters_2])
runtimes = np.concatenate([best_runtimes, best_runtimes_2])
runtimes_per_iteration = runtimes[runtimes > 0] / iterations[runtimes > 0]

N_tests_combined = N_tests_combined[runtimes > 0]

A2 = np.vstack([N_tests_combined, np.ones(N_tests_combined.size)]).T
out2 = np.linalg.lstsq(A2, runtimes_per_iteration)
m2, c2 = out2[0]
resid2 = out2[1][0] 

plt.figure()
plt.scatter(N_tests_combined, np.log(runtimes_per_iteration + 1e-30))
plt.plot(N_tests_combined, np.log(N_tests_combined * m2 + c2 + 1e-30))
plt.xlabel('N')
plt.ylabel('log( runtime per iteration)')
plt.title('Residual of linear fit {:3.2g} - runtime scales with N'.format(resid2))
plt.tight_layout()
plt.savefig('N_vs_runtime_per_iter.png')
plt.show()

