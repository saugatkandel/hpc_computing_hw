#!/bin/bash
#MSUB -N strong_scaling
#MSUB -A e30514
#MSUB -m e
#MSUB -l nodes=1:ppn=20
#MUSB -l walltime=00:20:00
#MSUB -q short
cd $PBS_O_WORKDIR
module load mpi
module load python/anaconda3.6
> runtimes_strong_scaling.dat
> strong_scaling_np.dat
for np in 1 5 10 15 20
do
echo $np >> strong_scaling_np.dat
mpirun -np $np ./duffing_parallel 0.5 100000 10000 100 test.out >> runtimes_strong_scaling.dat
done
python plot_runtimes_strong_scaling.py