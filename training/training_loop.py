import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from .generate_slurm import main as sample_fid_slurm
from .utils import DistributedSampleBank

@torch.no_grad()
def edm_sampler(
    net, noisy_images, class_labels=None, randn_like=torch.randn_like, base_sigma = None,
    sampleBank = None, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
        
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    

    # Time step discretization.
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device='cuda')
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    i = torch.nonzero(t_steps >= base_sigma, as_tuple=False).max().item()    
    t_steps = t_steps[i:]
    t_steps[0] = base_sigma
    noisy_images = noisy_images.cuda()    
    num_steps = len(t_steps) - 1
    x_next = noisy_images.to(torch.float64).cuda()
 
    # sampleBank.add_samples(x_next.detach(), torch.ones(x_next.size(0), device = 'cuda') * base_sigma)    
        
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
        # sampleBank.add_samples(x_next.detach(), torch.ones(x_next.size(0), device = 'cuda') * t_next)

    return x_next


#----------------------------------------------------------------------------

def training_loop(
    config,
    run,
    run_dir             = '.',      # Output directory.
    dataset_clean_kwargs      = {},       # Options for training set.
    dataset_cprt_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    loss_cprt_kwargs    = {},       # Options for loss checkpointing.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    fid_ref             = None,   # FID reference dataset for sampling fid
    seed                = 0,        # Global random seed.
    power               = 0.0,      # Noise power.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    ckpt_ticks          = 2,       # How often to save network checkpoints, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    last_n_snapshots_to_keep = 5,  # Number of snapshots to keep on disk
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):  
    
    if dataset_clean_kwargs is None:
        assert power == 0.0, "clean dataset is not provided, power should be 0.0"        
        
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    
    if dataset_clean_kwargs:
        dist.print0('Loading clean image dataset...')
        dataset_clean_obj = dnnlib.util.construct_class_by_name(**dataset_clean_kwargs)    
        dataset_clean_sampler = misc.InfiniteSampler(dataset=dataset_clean_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)    
        dataset_clean_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_clean_obj, sampler=dataset_clean_sampler, batch_size=batch_gpu, **data_loader_kwargs)) 
        torch.distributed.barrier()
    else:
        dist.print0('Clean image dataset is not provided. Set to None')
        dataset_clean_obj = None
        dataset_clean_sampler = None
        dataset_clean_iterator = None

    
    if dataset_cprt_kwargs:    
        dist.print0('Loading copyright dataset...')
        dataset_cprt_obj = dnnlib.util.construct_class_by_name(**dataset_cprt_kwargs) # subclass of training.dataset.Dataset
        dataset_cprt_sampler = misc.InfiniteSampler(dataset=dataset_cprt_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)    
        
        data_loader_kwargs['prefetch_factor']=None
        data_loader_kwargs['num_workers']=0
        
        dataset_cprt_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_cprt_obj, sampler=dataset_cprt_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    else:
        dist.print0('Copyright image dataset is not provided. Set to None')
        dataset_cprt_obj = None
        dataset_cprt_sampler = None
        dataset_cprt_iterator = None    
    
    eval_jobs = []
    
    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(
        img_resolution=dataset_clean_obj.resolution if dataset_clean_obj is not None else dataset_cprt_obj.resolution, 
        img_channels=dataset_clean_obj.num_channels if dataset_clean_obj is not None else dataset_cprt_obj.num_channels, 
        label_dim=dataset_clean_obj.label_dim if dataset_clean_obj is not None else dataset_cprt_obj.label_dim
    )    

    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    loss_fn_cprt = dnnlib.util.construct_class_by_name(**loss_cprt_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
                    
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dist.get_rank()])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    cur_tick = 0
    cur_nimg = 0

    if dataset_cprt_kwargs:  
        if dist.get_rank() == 0:
            sampleBank = DistributedSampleBank(max_samples = 50000)    
        else:
            sampleBank = None
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    torch.distributed.barrier() 
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=False)
        optimizer.load_state_dict(data['optimizer_state'])
        cur_nimg = data['nimg']
        cur_tick = data['tick']
       
        dist.print0(f"loading sampleBank from {os.path.join(os.path.dirname(resume_state_dump), 'sampleBank-last.pkl')}...")
        if dist.get_rank() == 0 and dataset_cprt_kwargs is not None:
            sampleBank.load(os.path.join(os.path.dirname(resume_state_dump), 'sampleBank-last.pkl'))                

        dist.print0(f'Resuming from tick {cur_tick}, cur_nimg {cur_nimg }')
        del data # conserve memory
                
    torch.distributed.barrier()
    
    
    if dataset_cprt_kwargs:
        # Broadcast the sample bank to all processes
        obj_list = [sampleBank]  # List to broadcast
        torch.distributed.broadcast_object_list(obj_list, src=0)
        sampleBank = obj_list[0]  # Extract the broadcasted object
    
    torch.distributed.barrier()     
    dist.print0('Construct sample bank...')   
    
    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    # cur_nimg = resume_kimg * 1000
    
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                
                ## Clean image dataset
                if dataset_clean_iterator is not None:
                    images, labels = next(dataset_clean_iterator)
                    images = images.to(device).to(torch.float32)
                    labels = labels.to(device)
                    loss = loss_fn(
                        net=ddp, 
                        images=images, 
                        labels=labels, 
                        augment_pipe=augment_pipe, 
                    ) 
                    if dataset_cprt_kwargs:
                        loss = loss * 0.5 
                        
                    training_stats.report('Loss/loss_clean', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                
        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            
        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
        
        if dist.get_rank() == 0:
            run.log(
                dict( 
                    tick = cur_tick,
                    kimg = cur_nimg / 1e3,
                    time = dnnlib.util.format_time(tick_end_time - start_time),
                    sec_per_tick = tick_end_time - tick_start_time,
                    sec_per_kimg = (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3,
                    maintenance = maintenance_time,
                    cpumem = psutil.Process(os.getpid()).memory_info().rss / 2**30,
                    gpumem = torch.cuda.max_memory_allocated(device) / 2**30,
                    reserved = torch.cuda.max_memory_reserved(device) / 2**30
                )
            )

            for job in eval_jobs:                        
                if job.done():
                    print(f"JOB {job.job_id} is done")
                    try:
                        run.log(job.result())
                    except Exception as e:
                        print(f"JOB {job.job_id} failed with error: {e}")
                        # breakpoint()
                    eval_jobs.remove(job)
                else:
                    print(f"JOB {job.job_id} is not done")
                
        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        save_snapshot = snapshot_ticks is not None and (done or cur_tick % snapshot_ticks == 0)
        save_ckpt = ckpt_ticks is not None and cur_tick % ckpt_ticks == 0
        # if ((snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0)) or (ckpt_ticks is not None and cur_tick % ckpt_ticks == 0):
        if save_snapshot or save_ckpt:
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe)
            if dataset_clean_kwargs:
                data |= dict(dataset_clean_kwargs = dict(dataset_clean_kwargs))
            if dataset_cprt_kwargs:
                data |= dict(dataset_cprt_kwargs = dict(dataset_cprt_kwargs))
            
            
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory                
                
            if dist.get_rank() == 0:
                if save_snapshot:
                    ckpt_dir = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                    with open(ckpt_dir, 'wb') as f:
                        pickle.dump(data, f)                                            
                    
                # Delete old network snapshots
                import glob, re
                snapshot_files = sorted(glob.glob(os.path.join(run_dir, 'network-snapshot-*.pkl')))
                snapshot_files = [f for f in snapshot_files  if re.match(r'.*network-snapshot-\d+\.pkl$', f)]                                    
                        
                dist.print0(f'NUM SNAPSHOT FILES: {len(snapshot_files)}')
                dist.print0(f"last_n_snapshots_to_keep: {last_n_snapshots_to_keep}")                
                
                if len(snapshot_files) > last_n_snapshots_to_keep:
                    for file in snapshot_files[:-last_n_snapshots_to_keep]:
                        os.remove(file)
                        dist.print0(f"Removed {file}")
                            
                if save_ckpt:
                    ckpt_dir = os.path.join(run_dir, f'network-ckpt-last.pkl')
                    with open(ckpt_dir, 'wb') as f:
                        pickle.dump(data, f)
                    dist.print0(f'Cur Tick [{cur_tick}]: Saved checkpoint to {ckpt_dir}')
                
            del data # conserve memory            
        
        # Save full dump of the training state.
        if save_ckpt:
            if dist.get_rank() == 0:
                if dataset_cprt_kwargs:
                    sampleBank.save(os.path.join(run_dir, f'sampleBank-last.pkl'))
                torch.save(
                    dict(net=net, optimizer_state=optimizer.state_dict(), nimg = cur_nimg, tick = cur_tick+1), 
                    os.path.join(run_dir, f'training-state-last.pt')
                )
                dist.print0(f'Cur Tick [{cur_tick}]: Saved training state to {os.path.join(run_dir, f"training-state-last.pt")}')

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            stats = dict(training_stats.default_collector.as_dict())
            stats |= {'cur_tick':cur_tick}               
                    
            run.log(stats)
            
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
