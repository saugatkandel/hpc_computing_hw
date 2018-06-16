#!/bin/bash
#MSUB -N run_n
#MSUB -A e30514
#MSUB -m e
#MSUB -l nodes=1:ppn=10
#MUSB -l walltime=00:20:00
#MSUB -q short
cd $PBS_O_WORKDIR
module load mpi
module load blas-lapack/3.6.0_gcc
module load python/anaconda3.6
> runtimes_n.dat
> runtimes_n_params.dat
for N in 50 100 150 200
do
echo $N >> runtimes_n_params.dat
mpirun -np 5 ./euler2d_parallel ${N} 50 2 test_runtime.out >> runtimes_n.dat
done
python plot_runtimes_n.py
