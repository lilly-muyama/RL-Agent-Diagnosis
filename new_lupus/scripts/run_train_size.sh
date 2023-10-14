#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=lillian.muyama@inria.fr  
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

#SBATCH --job-name=biopsy_9_with_missingness_levels
#SBATCH --nodes=1                
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=8 
#SBATCH --partition=cpu_devel        
#SBATCH --mem=128G               
#SBATCH --hint=multithread     
#SBATCH --array=0-4

echo "### Running $SLURM_JOB_NAME with array task $SLURM_ARRAY_TASK_ID ###"
SEEDS=(42 63 84 105 126)

python3 dqn_train_size.py --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} --size 0.01


