#!/bin/bash
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=lillian.muyama@inria.fr  
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

#SBATCH --job-name=missing_0.3_9_42
#SBATCH --nodes=1                
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4 
#SBATCH --partition=cpu_devel        
#SBATCH --mem=64G               
#SBATCH --hint=multithread     
#SBATCH --array=0-4

python3 dqn_missingness.py --seed 84 --beta 9 --missingness 0.2

