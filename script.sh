#!/bin/bash
##### Example for 1 node and 18 tasks 0 GPU #####
#SBATCH -J test_tn-torch_cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --threads-per-core=1
#SBATCH --time=12:00:00
# SBATCH --mail-user=ponnaganti@irsamc.ups-tlse.fr
#SBATCH --out=myJob-cpu-%j.out
#SBATCH --err=myJob-cpu-%j.err
# --qos=<qos_name>

# in principle, you can request up to 4 GPUs, by e.g. --gres=gpu:4
# They will be available at cuda ids: cuda:0 up to cuda:3
#
# If you want to request less than 4, you have to specify also CPUs & the memory

module purge
module load miniconda/4.9.2
source /usr/local/miniconda/4.9.2/etc/profile.d/conda.sh
conda activate pytorch-1.8.0

export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cores=$OMP_NUM_THREADS

python april22.py >out &
