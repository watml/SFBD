import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--cprt_data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default=None, required=False)
@click.option('--clean_data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default=None, required=False)
@click.option('--fid_ref',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default=None, required=False)
@click.option('--base_sigma',          help='base sigma', metavar='FLOAT',                     type=float, required=True)
@click.option('--base_noise_seed',          help='Random seed  [default: 0]', metavar='INT',              type=int, default=0)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)
@click.option('--power',         help='Power of the variance', metavar='FLOAT',                     type=float, default=0.0, show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=0), default=1, show_default=True)


# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--disc_suffix',   help='Suffix string to include in result dir name', metavar='STR',        type=str)
# @click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--rerun_samedir',      help='Use the same dir when rerun',                           is_flag=True, default=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--ckpt_ticks',          help='How often to dump ckpts', metavar='TICKS',                   type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--last_n_snapshots_to_keep',          help='Keep last n snapshots', metavar='INT',                   type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

    
    
def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    assert opts.cond == 0, 'Conditional training is not supported in this version of the code.'
    # assert opts.augment == 0, 'Augmentation is not supported in this version of the code.'
    assert opts.clean_data is not None or  opts.cprt_data is not None , 'At least one dataset must be specified.'    
        
    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.last_n_snapshots_to_keep = opts.last_n_snapshots_to_keep
    c.fid_ref = opts.fid_ref
    c.power=opts.power
    c.ckpt_ticks = opts.ckpt_ticks
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.loss_cprt_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
   
    # Validate dataset options.    
    if opts.clean_data is None:
        dist.print0('\n' * 3)
        dist.print0('!' * 25)
        dist.print0('No clean dataset specified. Training will be performed on the copyright dataset.')
        dist.print0('\n' * 3)
        c.dataset_clean_kwargs = None
    else:
        dist.print0('\n' * 3)
        dist.print0(f'Loading clean dataset: {opts.clean_data}')
        c.dataset_clean_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset', 
            path=opts.clean_data, 
            use_labels=opts.cond, 
            xflip=opts.xflip, 
            cache=opts.cache
        )
        try:
            dataset_clean_obj = dnnlib.util.construct_class_by_name(**c.dataset_clean_kwargs)
            dataset_name = dataset_clean_obj.name
            c.dataset_clean_kwargs.resolution = dataset_clean_obj.resolution # be explicit about dataset resolution
            c.dataset_clean_kwargs.max_size = len(dataset_clean_obj) # be explicit about dataset size
            if opts.cond and not dataset_clean_obj.has_labels:
                raise click.ClickException('--cond=True requires labels specified in dataset.json')
            del dataset_clean_obj # conserve memory
        except IOError as err:
            raise click.ClickException(f'--clean_data: {err}')
        dist.print0('\n' * 3)
        
        dataset_kwargs = c.dataset_clean_kwargs
    
    if opts.cprt_data is None:
        c.dataset_cprt_kwargs = None
    else:
        raise click.ClickException('Copyright dataset is not supported in this version of the code.')    

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.    
    assert opts.precond  == 'edm', 'Only EDM is supported in this version of the code.'    
    
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'        
        c.loss_cprt_kwargs.class_name = 'training.loss.EDMLoss'     

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
       
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    # desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-batch{c.batch_size:d}'    
    desc += f'-var_power-{str(c.power).replace(".", "_").replace("-", "n")}' if c.power != 0.0 else ''
    desc += f"-{opts.disc_suffix:s}-{dtype_str:s}"
    
    if c.dataset_cprt_kwargs is None or c.dataset_clean_kwargs is None:
        if c.dataset_clean_kwargs is None:
            desc += f'-cprt-only'
        if c.dataset_cprt_kwargs is None:
            desc += f'-clean-only'
    
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Use the same directory when rerunning 
    run_dir = os.path.join(opts.outdir, desc)     
    if dist.get_rank() == 0 :
        c.run_dir = run_dir
        import pathlib
        pathlib.Path(c.run_dir).mkdir(parents=True, exist_ok=True)
        assert os.path.exists(c.run_dir)
    else:
        c.run_dir = None
    
    resume_pkl = os.path.join(run_dir, f'network-ckpt-last.pkl')
    resume_pt = os.path.join(run_dir, f'training-state-last.pt')
    if os.path.exists(resume_pkl) and os.path.exists(resume_pt):        
        dist.print0('-'*20)
        dist.print0(f'\n Resuming from: \n resume_pkl: {resume_pkl}\n  resume_pt: {resume_pt}')
        dist.print0('-'*20)
        c.resume_pkl = resume_pkl
        c.resume_state_dump = resume_pt
        match = re.fullmatch(r'network-snapshot-(\d+).pkl', os.path.basename(resume_pkl))    
    else:
        dist.print0(f'No resume files found in {c.run_dir}')
            

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f"Clean Dataset path:      {c.dataset_clean_kwargs.path if c.dataset_clean_kwargs is not None else 'No clean dataset specified.'}")
    dist.print0(f"Copyright Dataset path:  {c.dataset_cprt_kwargs.path if c.dataset_cprt_kwargs is not None else 'No copyright dataset specified.'}")    
    dist.print0(f'fid_ref:                 {c.fid_ref}')
    dist.print0(f'Class-conditional:       {dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    if dist.get_rank() == 0:
        import wandb
        run = wandb.init(
            project=desc,
            config=c
        )
    else:
        import wandb
        run = None
    training_loop.training_loop(config = c, run=run, **c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
