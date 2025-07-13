#!/bin/bash
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=SLURM_PARTITION
#SBATCH --output=denoise_logs/%x.%j.log
#SBATCH --export=ALL
#SBATCH --qos=SLURM_QOS
#SBATCH --signal=B:USR1@60

num_gpus=$(nvidia-smi --list-gpus | wc -l)

echo "Running on $num_gpus GPUs"
echo "NODE: $(hostname)"

START=$1
END=$2
NETWORK=$3
CPRT_DATA=$4
BASE_SIGMA=$5
ITER=$6
OUTDIR=$7

echo "START: $START"
echo "END: $END"
echo "NETWORK: $NETWORK"
echo "CPRT_DATA: $CPRT_DATA"
echo "BASE_SIGMA: $BASE_SIGMA"
echo "ITER: $ITER"
ecgi "OUTDIR: $OUTDIR"

python denoise.py \
    --steps=40 \
    --network=$NETWORK \
    --cprt_data=$CPRT_DATA \
    --power=0.0 \
    --start=$START \
    --end=$END \
    --base_sigma=$BASE_SIGMA \
    --seeds=0-9999 \
    --outdir=$OUTDIR 
