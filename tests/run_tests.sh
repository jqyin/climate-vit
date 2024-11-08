#!/bin/bash

image=nersc/pytorch:24.06.01
env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_24.06.01
tp=1
cp=1

# parse args
for arg in "$@"
do
    if [[ $arg == tp=* ]]; then
        tp="${arg#*=}"
    elif [[ $arg == cp=* ]]; then
        cp="${arg#*=}"
    fi
done

ngpu=$(( ${tp} * ${cp} ))
export MASTER_ADDR=$(hostname)
srun --nodes 1 --ntasks-per-node $ngpu --gpus-per-node $ngpu -u shifter --image=$image --module=gpu,nccl-plugin --env PYTHONUSERBASE=$env \
    bash -c "
    source export_DDP_vars.sh
    export TP=${tp}
    export CP=${cp}
    python -m pytest -s tests/test_distributed.py
    "
