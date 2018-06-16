#!/bin/bash
#MSUB -N run_mn
#MSUB -A e30514
#MSUB -m e
#MSUB -l nodes=1:ppn=10
#MUSB -l walltime=00:20:00
#MSUB -q short
cd $PBS_O_WORKDIR
module load mpi
module load python/anaconda3.6
> runtimes_mn.dat
> runtimes_mn_params.dat
for M in 5000 10000 15000 20000 25000
do
for N in 5000 10000 15000 20000 25000
do
echo $M $N >> runtimes_mn_params.dat
mpirun -np 5 ./duffing_parallel 0.5 ${M} ${N} 100 test.out >> runtimes_mn.dat
done
done
python plot_runtimes_mn.py
