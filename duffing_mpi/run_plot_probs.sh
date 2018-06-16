#!/bin/bash
#MSUB -N run_prob
#MSUB -A e30514
#MSUB -m e
#MSUB -l nodes=1:ppn=5
#MUSB -l walltime=00:20:00
#MSUB -q short
cd $PBS_O_WORKDIR
module load mpi
module load python/anaconda3.6
mpirun -np 5 ./duffing_parallel 0.5 10000 10000
python plot_probability.py