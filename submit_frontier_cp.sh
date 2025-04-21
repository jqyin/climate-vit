#!/bin/bash -l
#SBATCH -J test
#SBATCH -A stf218
#SBATCH -t 00:50:00
#SBATCH -N 8
#SBATCH -q debug
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -o vit-patch8.o%j
#SBATCH -e vit-patch8.e%j

LOGDIR=$PWD/logs
mkdir -p ${LOGDIR}
args="${@}"
echo $args
emb=1024
patch=1
cp=64
tp=1

export HDF5_USE_FILE_LOCKING=FALSE
module use /sw/aaims/crusher/modulefiles
module load xforge
#source env.sh

#export NCCL_DEBUG=INFO
#export NCCL_ALGO=Tree
export FI_PROVIDER=cxi
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RDZV_PROTO=alt_read
export NCCL_CROSS_NIC=1
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=disabled

#MODEL
for algo in ring ringX
do
  rm -rf ./checkpoints
  args="--config=mp_emb${emb}_${algo} --tensor_parallel=$tp --context_parallel=$cp --parallel_order=tp-cp-dp"
  HOME=/tmp srun --nodes=${SLURM_NNODES} \
     --network=disable_rdzv_get \
     --ntasks=$((SLURM_NNODES*8)) \
     --gpu-bind=closest -c7  \
     bash -c "source export_DDP_vars.sh; python -u train_mp.py ${args}"  2>&1 | tee log.emb${emb}.n${SLURM_NNODES}.${algo}.tp${tp}.cp${cp}
done 

#patch
for algo in megatron ring ringX
do
  rm -rf ./checkpoints
  args="--config=mp_p${patch}_${algo} --tensor_parallel=$tp --context_parallel=$cp --parallel_order=tp-cp-dp"
  HOME=/tmp srun --nodes=${SLURM_NNODES} \
     --network=disable_rdzv_get \
     --ntasks=$((SLURM_NNODES*8)) \
     --gpu-bind=closest -c7  \
     bash -c "source export_DDP_vars.sh; python -u train_mp.py ${args}"  2>&1 | tee log.p${patch}.n${SLURM_NNODES}.${algo}.tp${tp}.cp${cp}
done 
 
