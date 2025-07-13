#!/bin/bash

NETWORK="PATH TO THE MODEL CHECKPOINT"  # Replace with the actual path to your network checkpoint
CPRT_DATA="PATH TO THE CPRT DATA"  # Replace with the actual path to your copyrighted data
BASE_SIGMA=1.3824   # Noise Level used for corrupting the data
ITER= # The iteration number for the model, e.g., 1, 2, 3, etc. 
DIR_TO_DENOISED_IMAGES="PATH TO THE DIRECTORY WHERE DENOISED IMAGES WILL BE SAVED"  # Replace with the actual path to your output directory
NUM_TASKS=5 # Number of tasks to split the workload into

OUTDIR=$DIR_TO_DENOISED_IMAGES/celeba_${BASE_SIGMA}_iter${ITER}/out

NUM_IMAGES=$(find $CPRT_DATA -type f -name "*.png" | wc -l)
echo "Number of images: $NUM_IMAGES"

IMAGES_PER_TASK=$((NUM_IMAGES / NUM_TASKS))
echo "Images per task: $IMAGES_PER_TASK"

mkdir -p $OUTDIR  # Create the OUTDIR directory if it doesn't exist

for i in $(seq 0 $((NUM_TASKS-1)))
do
    START=$((i * IMAGES_PER_TASK))
    END=$(((i + 1) * IMAGES_PER_TASK))
    echo "Submitting task $i: $START-$END"
    sbatch scripts/denoise.sh $START $END $NETWORK $CPRT_DATA $BASE_SIGMA $ITER $OUTDIR
    sleep 1
done
