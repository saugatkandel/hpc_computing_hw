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
> runtimes_weak_scaling.dat
> weak_scaling_params.dat
export N=100000
for np in 1 5 10 15 20
do
export M=${np}000
echo $np $M $N >> weak_scaling_params.dat
mpirun -np $np ./duffing_parallel 0.5 $M $N 100 test.out >> runtimes_weak_scaling.dat
done
python plot_runtimes_weak_scaling.py