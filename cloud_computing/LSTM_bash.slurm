#!/bin/bash
#SBATCH -J "train_st7_LSTM_RAWDATA31_v4"
#SBATCH -A igdino
#SBATCH -p barbun-cuda
# SBATCH -p debug  
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --gres=gpu:1
# SBATCH --time=00:15:00
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out 
#SBATCH --error=slurm-%j.err

#Setup environment
module purge
eval "$(/truba/sw/centos7.9/lib/anaconda3/2023.03/bin/conda shell.bash hook)"
conda activate pytorch_igd

# Disable NVML to avoid NVLink warnings
export TORCH_CUDA_NVML_DISABLED=1

# Verify PyTorch and CUDA
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
print('Current CUDA device:', torch.cuda.current_device())
print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
"

# Log job information
cat <<EOT >> process.txt
Job Name: train_st7_LSTM_RAWDATA31_v4
Account: igdino
Queue: $SLURM_JOB_PARTITION
Nodes: $SLURM_JOB_NODELIST
Cores per node: $SLURM_CPUS_PER_TASK
Job ID: $SLURM_JOB_ID
Job start time: $(date)

Running Python script: train_st7_LSTM_RAWDATA31_v4.py
EOT

# Run Python script
python train_st7_LSTM_RAWDATA31_v4.py

# Log module information and server name
echo "We have the modules: $(module list 2>&1)" > ${SLURM_JOB_ID}.info
echo "Server name: $(hostname)" >> process.txt

# Record job end time
echo "Job end time: $(date)" >> process.txt

