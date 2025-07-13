CLEAN_DATA="/path/to/clean_dataset"   # Absolute path to your copyright-free data
TRANSFER=None                         # Pre-training: keep None; fine-tuning: set to the checkpoint from the previous run

num_gpus=$(nvidia-smi --list-gpus | wc -l)

echo "Running on $num_gpus GPUs"

echo "GPU models:"
for ((i=0; i<$num_gpus; i++))
do
    gpu_model=$(nvidia-smi --list-gpus | sed -n "$((i+1))p")
    echo "GPU $i model: $gpu_model"
done

torchrun --standalone --nproc_per_node=$num_gpus train.py --outdir=training-runs-pretrain \
    --clean_data=$CLEAN_DATA \
    --arch=ncsnpp \
    --batch=256 \
    --cres=1,2,2,2 \
    --lr=2e-4 \
    --dropout=0.05 \
    --augment=0.15 \
    --power=0.0 \
    --base_sigma=0 \
    --snap=25 \
    --last_n_snapshots_to_keep=15 \
    --cond=0 \
    --workers=1 \
    --disc_suffix='celeba'
