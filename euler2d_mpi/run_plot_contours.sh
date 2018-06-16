#!/bin/bash
#MSUB -N run_cont
#MSUB -A e30514
#MSUB -m e
#MSUB -l nodes=1:ppn=5
#MUSB -l walltime=00:20:00
#MSUB -q short
cd $PBS_O_WORKDIR
module load mpi
module load blas-lapack/3.6.0_gcc
module load python/anaconda3.6
mpirun -np 5 ./euler2d_parallel 100 100
python plot_contours.py
