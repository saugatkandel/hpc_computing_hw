#!/bin/bash
#MSUB -N strong_scaling
#MSUB -A e30514
#MSUB -m e
#MSUB -l nodes=1:ppn=20
#MUSB -l walltime=00:20:00
#MSUB -q short
cd $PBS_O_WORKDIR
module load mpi
module load blas-lapack/3.6.0_gcc
module load python/anaconda3.6
> runtimes_weak_scaling.dat
> weak_scaling_params.dat
for np in 1 5 10 15 20
do
export N=$(($np * 50))
echo $np $N >> weak_scaling_params.dat
mpirun -np $np ./euler2d_parallel $N 10 2 test_weak_scaling.out >> runtimes_weak_scaling.dat
done
python plot_runtimes_weak_scaling.py
