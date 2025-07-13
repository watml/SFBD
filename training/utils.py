# prerequisites
import torch
import os
import numpy as np
import torch.nn as nn
from torch_utils import distributed as dist
import pickle
import torch.multiprocessing as mp
import random

class DistributedSampleBank:
    def __init__(self, max_samples):
        """
        Initialize the Distributed Sample Bank.

        Args:
            max_samples (int): Maximum number of samples to store in the bank.
        """
        self.max_samples = max_samples
        self.samples = mp.Manager().list()  # Shared memory for samples (numpy arrays)
        self.labels = mp.Manager().list()   # Shared memory for labels (numpy arrays)

    def add_samples(self, x, t):
        """
        Add samples to the bank.

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W] on GPU.
            t (torch.Tensor): Tensor of shape [B] on GPU.
        """
        x_cpu = x.detach().cpu().numpy()  # Convert to NumPy array on CPU
        t_cpu = t.detach().cpu().numpy()  # Convert to NumPy array on CPU

        # Add samples and labels to the bank
        for i in range(x_cpu.shape[0]):
            self.samples.append(x_cpu[i])
            self.labels.append(t_cpu[i])

        # Ensure the bank size doesn't exceed max_samples
        while len(self.samples) > self.max_samples:
            self.samples.pop(0)  # Remove the oldest sample
            self.labels.pop(0)  # Remove the corresponding label

    def get_samples(self, B):
        """
        Fetch random samples from the bank.

        Args:
            B (int): Number of samples to fetch.

        Returns:
            x (torch.Tensor): Tensor of shape [B, C, H, W].
            t (torch.Tensor): Tensor of shape [B].
        """
        if len(self.samples) == 0:
            raise ValueError("The bank is empty. Add samples first.")

        indices = random.sample(range(len(self.samples)), min(B, len(self.samples)))
        x = np.stack([self.samples[i] for i in indices])
        t = np.array([self.labels[i] for i in indices])
        return torch.tensor(x), torch.tensor(t)

    def get_num_samples(self):
        """
        Return the current number of samples in the bank.

        Returns:
            int: Number of samples currently stored in the bank.
        """
        return len(self.samples)

    def save(self, filepath):
        """
        Save the state of the bank to a file.

        Args:
            filepath (str): Path to save the bank.
        """
        state = {
            "samples": list(self.samples),
            "labels": list(self.labels),
            "max_samples": self.max_samples,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load(self, filepath):
        """
        Load the state of the bank from a file.

        Args:
            filepath (str): Path to load the bank.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.samples = mp.Manager().list(state["samples"])
        self.labels = mp.Manager().list(state["labels"])
        self.max_samples = state["max_samples"]

    

def create_mollifier(dataset_obj, dataset_root, mollifier_ckpt = None, power = 0.0, cache = False, seed = 0):
    if mollifier_ckpt is not None:
        dist.print0("mollifier_ckpt is not None")
        dist.print0("\n" * 5)
        
        mollifier = AdaptMollificationNoise(
            image_shape=(dataset_obj.num_channels , dataset_obj.resolution, dataset_obj.resolution), 
            complex_weight=True if 'complex' in mollifier_ckpt else False,
            random_seed=seed,
        )
        
        msg = f"Rank {dist.get_rank()}: Loading mollifier from {mollifier_ckpt}"
        if 'complex' in mollifier_ckpt:
            msg += " with complex weight"
        else:
            msg += " with real weight"
        
        print(msg)
        
        mollifier.load_state_dict(torch.load(mollifier_ckpt, map_location='cuda'))
        mollifier.eval()
        for p in mollifier.parameters():
            p.require_grads = False
            
        for p in mollifier.hermSymPara.parameters():
            p.require_grads = False
            
        print(f"Rank {dist.get_rank()}: Loaded mollifier from {mollifier_ckpt}")        
        # mollifier = torch.compile(mollifier)
    else:
        dist.print0("mollifier_ckpt is None")
        dist.print0(f"Using mollifier with power {power}")
        dist.print0("\n" * 5)
        mollifier = Mollification(dataset=dataset_obj, var_power=power, dataset_root=dataset_root, cache = cache)             
    
    return mollifier

class Mollification():
    def __init__(self, dataset = None, var_power = 0.0, dataset_root = './', batch_size = 32, num_workers = 4, cache = False, random_seed = 0):
        if dataset is not None:
            self.dataset = dataset

        self.dataset_root = dataset_root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.var_power = var_power
        self.cache = cache
        self.generator_numpy = np.random.default_rng(random_seed)
        self.generator_torch = torch.Generator()
        self.generator_torch.manual_seed(random_seed)
    
    def data_transform(self, X):
 
        X = 2 * X - 1.0
        return X

    def get_freq_var(self):
        cache_filename = f"var_tensor.npy"
        if os.path.exists(os.path.join(self.dataset_root, cache_filename)) and self.cache:
            print(f"Loading variance tensor from file: {cache_filename}")
            return np.load(os.path.join(self.dataset_root, cache_filename))

        if not hasattr(self, 'dataset'):
            raise ValueError('Please provide the dataset to compute the variance of the frequency of the dataset')

        dataset = self.dataset

        # compute the variance of the frequency of the dataset
        train_loader_ = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        ### Compute the mean the dataset
        mean_tensor = None  # Mean for each channel
        num_samples = 0

        from tqdm import tqdm
        # Example of iterating through the DataLoader
        for i, sample in tqdm(enumerate(train_loader_)):
            if len(sample) > 1:
                images, y = sample
            else:
                images = sample

            images = self.data_transform(images).numpy()
            images = np.fft.rfft2(images, norm = 'forward')
            images = np.stack([np.real(images), np.imag(images)], axis = -1)

            if mean_tensor is None:
                mean_tensor = np.zeros_like(images[0])           

            batch_size = images.shape[0]

            # Update the running mean
            mean_tensor = (mean_tensor * num_samples + images.sum(axis=0)) / (num_samples + batch_size)
            
            # Update the total number of samples
            num_samples += batch_size   

        ### Compute the variance the dataset
        var_tensor = None  # Mean for each channel
        num_samples = 0

        for i, sample in tqdm(enumerate(train_loader_)):
            if len(sample) > 1:
                images, y = sample
            else:
                images = sample

            images = self.data_transform(images).numpy()
            images = np.fft.rfft2(images, norm = 'forward')
            images = np.stack([np.real(images), np.imag(images)], axis = -1)

            if var_tensor is None:
                var_tensor = np.zeros_like(images[0])           

            batch_size = images.shape[0]
            var_tensor = (var_tensor * num_samples + ((images - mean_tensor[None, :]) ** 2).sum(axis = 0) )  / (num_samples + batch_size)
            
            # Update the total number of samples
            num_samples += batch_size

        del train_loader_
        np.save(os.path.join(self.dataset_root, f"mean_tensor.npy"), mean_tensor)
        np.save(os.path.join(self.dataset_root, f"var_tensor.npy"), var_tensor)
              
        return var_tensor            

    def set_data_freq_var(self):
        # redistribute the variance to the frequencies' real and imaginary parts
        
        if self.var_power >= 0.0:
            data_freq_var_ = 0.5 * self.get_freq_var().sum(axis = -1).mean(axis = 0) ** self.var_power
        else:
            data_freq_var_ = self.get_freq_var().sum(axis = -1).mean(axis = 0)
            data_freq_var_[data_freq_var_ > 0] = data_freq_var_[data_freq_var_ > 0] ** self.var_power            
            data_freq_var_ = 0.5 * data_freq_var_
            
        H = data_freq_var_.shape[0]
        data_freq_var_ = np.stack([data_freq_var_, data_freq_var_], axis = -1)

        data_freq_var_[0,0,-1] = 0
        data_freq_var_[0,-1,-1] = 0
        data_freq_var_[H//2,0,-1] = 0
        data_freq_var_[H//2,-1,-1] = 0

        data_freq_var_[0,0,-2] *= 2
        data_freq_var_[0,-1,-2] *= 2
        data_freq_var_[H//2,0,-2] *= 2
        data_freq_var_[H//2,-1,-2] *= 2
        import copy
        self.data_freq_var = copy.deepcopy(data_freq_var_)
        data_freq_var_[data_freq_var_ == 0] = 1.0
        self.data_freq_var_inv = 1 / data_freq_var_
        self.data_freq_var_inv[ data_freq_var_ == 0] = 0

    def mollify(self, x, fwd = True):
        if self.var_power == 0.0:
            return x
        if type(x) == np.ndarray:
            return self.__mollify_np__(x, fwd)
        elif type(x) == torch.Tensor:
            return self.__mollify_torch__(x, fwd)
        else:
            raise ValueError('Input should be either a numpy array or a torch tensor')


    def __mollify_torch__(self, x, fwd : bool):
        assert x.ndim == 4, f'Input tensor should have 4 dimensions. The input tensor has {x.ndim} dimensions'
        assert x.shape[2] == x.shape[3], f'Input tensor should have the same height and width. The input tensor has height {x.shape[2]} and width {x.shape[3]}'
        assert type(x) == torch.Tensor, f'Input tensor should be a torch tensor. The input tensor is of type {type(x)}'

        # if x.device != torch.device('cpu'):
        #     breakpoint()
        
        B, C, H, W = x.shape
        freq = torch.fft.rfft2(x, s=(H, W), norm='forward')
        freq = torch.view_as_real(freq)

        if not hasattr(self, 'data_freq_var_torch'):
            self.set_data_freq_var()
            self.data_freq_var_torch = torch.from_numpy(self.data_freq_var).to(x.device).to(x.dtype)
            self.data_freq_var_inv_torch = torch.from_numpy(self.data_freq_var_inv).to(x.device).to(x.dtype)
    
        if not hasattr(self, 'unit_spatial_std_torch'):
            if not hasattr(self, 'unit_spatial_std'):
                unit_spatial_std = np.fft.irfft2(self.data_freq_var[..., -2], s = (H, W))[0,0] ** 0.5
                self.unit_spatial_std =  unit_spatial_std
            self.unit_spatial_std_torch = torch.tensor(self.unit_spatial_std).to(x.device).to(x.dtype)
        
        if fwd:
            if self.unit_spatial_std_torch.device != x.device:
                self.unit_spatial_std_torch = self.unit_spatial_std_torch.to(x.device)
            if self.data_freq_var_torch.device != x.device:
                self.data_freq_var_torch = self.data_freq_var_torch.to(x.device)           

                      
            freq_mollified = freq * ( self.data_freq_var_torch[None, None, :] ** 0.5 / self.unit_spatial_std_torch)

        else:
            if self.unit_spatial_std_torch.device != x.device:
                self.unit_spatial_std_torch = self.unit_spatial_std_torch.to(x.device)
            if self.data_freq_var_inv_torch.device != x.device:
                self.data_freq_var_inv_torch = self.data_freq_var_inv_torch.to(x.device) 

            freq_mollified = freq *  self.data_freq_var_inv_torch[None, None, :] ** 0.5 * self.unit_spatial_std_torch

        x_mollified = torch.fft.irfft2(torch.view_as_complex(freq_mollified), s=(H, W), norm='forward')


        return x_mollified

    @torch.no_grad()
    def __mollify_np__(self, x, fwd : bool):
        assert x.ndim == 4, f'Input tensor should have 4 dimensions. The input tensor has {x.ndim} dimensions'
        assert x.shape[2] == x.shape[3], f'Input tensor should have the same height and width. The input tensor has height {x.shape[2]} and width {x.shape[3]}'
        assert type(x) == np.ndarray, f'Input tensor should be a numpy array. The input tensor is of type {type(x)}'
        
        if not hasattr(self, 'unit_spatial_std'):
            unit_spatial_std = np.fft.irfft2(self.data_freq_var[..., -2], s = (H, W))[0,0] ** 0.5
            self.unit_spatial_std =  unit_spatial_std

        B, C, H, W = x.shape
        freq = np.fft.rfft2(x, s=(H, W), norm = 'forward')
        freq = np.stack([np.real(freq), np.imag(freq)], axis = -1)    
        if fwd:    
            freq_mollified = freq * ( self.data_freq_var[None, None, :] ** 0.5 / self.unit_spatial_std )
        else:
            freq_mollified = freq *  self.unit_spatial_std * self.data_freq_var_inv[None, None, :] ** 0.5 

        freq_mollified = freq_mollified[..., 0] + 1j * freq_mollified[..., 1]   
        x_mollified = np.fft.irfft2(freq_mollified, s=(H, W), norm='forward') 
        return x_mollified
    
    @torch.no_grad()
    def getNoise(self, x):
        B, C, H, W = x.shape

        assert H == W, f'Input tensor should have the same height and width. The input tensor has height {H} and width {W}'

        if self.var_power == 0.0:
            return torch.randn( (*x.shape,), generator = self.generator_torch).to(x.device)

        if not hasattr(self, 'data_freq_var'):           
            self.set_data_freq_var()

        if not hasattr(self, 'unit_spatial_std'):
            unit_spatial_std = np.fft.irfft2(self.data_freq_var[..., -2], s = (H, W))[0,0] ** 0.5
            self.unit_spatial_std =  unit_spatial_std

        assert x.ndim == 4, f'Input tensor should have 4 dimensions. The input tensor has {x.ndim} dimensions'
        _, _, H, W = x.shape        

        noise = self.mollify(self.generator_numpy.normal(size = (B, C, H, W)))

        return torch.from_numpy(noise).to(x.device).to(x.dtype)
    
    
# class HermitianSymmetricParamMultiChannel(nn.Module):
#     def __init__(self, shape, num_channels=3):
#         """
#         Create a parameter tensor with Hermitian symmetry for multi-channel inputs.
#         Args:
#             shape (tuple): Shape of the original real-valued input (H, W).
#             num_channels (int): Number of channels (default: 3).
#         """
#         super().__init__()
#         self.H, self.W = shape
#         self.C = num_channels
#         self.fft2_shape = (self.C, self.H, self.W)

#         # Parametrize the unique part of the frequency domain
#         self.real = nn.Parameter(torch.rand(*self.fft2_shape))  # Real part
#         self.imag = nn.Parameter(torch.rand(*self.fft2_shape))  # Imaginary part
        
#     def __get_symmetric__(self, a):
#         # return a matrix with Hermitian symmetry
#         # a_flipped[..., i, j] = a[..., -i, -j]        
#         a_flipped = torch.flip(a, dims=(-2, -1))
#         a_flipped = a_flipped.roll(shifts=1, dims=-1).roll(shifts=1, dims=-2)
#         return a_flipped        

#     def forward(self):
#         """
#         Reconstruct the full tensor with Hermitian symmetry.
#         Returns:
#             torch.Tensor: Hermitian symmetric tensor of shape (C, H, W // 2 + 1).
#         """
#         # Clone the real and imaginary parts
#         real = self.real.clone()
#         imag = self.imag.clone()

#         # Enforce constraints on special frequencies
#         imag[:, 0, 0] = 0  # DC component is purely real
        
#         if self.W % 2 == 0:
#             # imag[:, :, -1] = 0  # Nyquist frequency is purely real
#             Nyqst = self.W // 2
#             imag[:, 0, Nyqst] = 0  # Nyquist frequency is purely real
#             imag[:, Nyqst, 0] = 0
#             imag[:, Nyqst, Nyqst] = 0
            
#         # Combine real and imaginary parts into a complex tensor
#         tensor = torch.complex(real, imag)

#         # Flip rows and enforce symmetry
#         flipped_real = self.__get_symmetric__(real)  # Flip rows
#         flipped_imag = -self.__get_symmetric__(imag)  # Flip rows and negate imaginary part

#         # Average the original and flipped values to ensure symmetry
#         tensor = 0.5 * tensor + 0.5 * torch.complex(flipped_real, flipped_imag)
#         return tensor


class ComplexHermitianSymmetricParamMultiChannel(nn.Module):
    def __init__(self, shape, num_channels=3):
        """
        Create a parameter tensor with Hermitian symmetry for multi-channel inputs.
        Args:
            shape (tuple): Shape of the original real-valued input (H, W).
            num_channels (int): Number of channels (default: 3).
        """
        super().__init__()
        self.H, self.W = shape
        self.C = num_channels
        self.fft2_shape = (self.C, self.H, self.W)

        # Parametrize the unique part of the frequency domain
        self.real = nn.Parameter(torch.zeros(*self.fft2_shape))  # Real part
        self.imag = nn.Parameter(torch.randn(*self.fft2_shape) / (self.H ** 2) )  # Imaginary part
        
    def __get_symmetric__(self, a):
        # return a matrix with Hermitian symmetry
        # a_flipped[..., i, j] = a[..., -i, -j]        
        a_flipped = torch.flip(a, dims=(-2, -1))
        a_flipped = a_flipped.roll(shifts=1, dims=-1).roll(shifts=1, dims=-2)
        return a_flipped        

    def forward(self):
        """
        Reconstruct the full tensor with Hermitian symmetry.
        Returns:
            torch.Tensor: Hermitian symmetric tensor of shape (C, H, W // 2 + 1).
        """
        # Clone the real and imaginary parts
        real = self.real.clone().sigmoid()
        imag = self.imag.clone().tanh()
        
        
        mask = torch.ones_like(imag)
        mask[..., 0, 0] = 0  # DC component is purely real        
        if self.W % 2 == 0:
            Nyqst = self.W // 2
            mask[..., 0, Nyqst] = 0  # Nyquist frequency is purely real
            mask[..., Nyqst, 0] = 0
            mask[..., Nyqst, Nyqst] = 0

        imag = imag * mask            
            
        # Combine real and imaginary parts into a complex tensor
        tensor = torch.complex(real, imag)

        # Flip rows and enforce symmetry
        flipped_real = self.__get_symmetric__(real)  # Flip rows
        flipped_imag = -self.__get_symmetric__(imag)  # Flip rows and negate imaginary part

        # Average the original and flipped values to ensure symmetry
        tensor = 0.5 * tensor + 0.5 * torch.complex(flipped_real, flipped_imag)
        return tensor
        
    
class HermitianSymmetricParamMultiChannel(nn.Module):
    def __init__(self, shape, num_channels=3):
        """
        Create a parameter tensor with Hermitian symmetry for multi-channel inputs.
        Args:
            shape (tuple): Shape of the original real-valued input (H, W).
            num_channels (int): Number of channels (default: 3).
        """
        super().__init__()
        self.H, self.W = shape
        self.C = num_channels
        self.fft2_shape = (self.C, self.H, self.W)

        # Parametrize the unique part of the frequency domain
        self.real = nn.Parameter(torch.zeros(*self.fft2_shape))  # Real part

    def __get_symmetric__(self, a):
        # return a matrix with Hermitian symmetry
        # a_flipped[..., i, j] = a[..., -i, -j]        
        a_flipped = torch.flip(a, dims=(-2, -1))
        a_flipped = a_flipped.roll(shifts=1, dims=-1).roll(shifts=1, dims=-2)
        return a_flipped        

    def forward(self):
        """
        Reconstruct the full tensor with Hermitian symmetry.
        Returns:
            torch.Tensor: Hermitian symmetric tensor of shape (C, H, W // 2 + 1).
        """
        # Clone the real part
        real = self.real.clone()
            
        # Flip rows and enforce symmetry
        flipped_real = self.__get_symmetric__(real)  # Flip rows

        # Average the original and flipped values to ensure symmetry
        tensor = 0.5 * real + 0.5 * flipped_real

        return tensor.sigmoid()    
    
    
class AdaptMollificationNoise(torch.nn.Module):
    def __init__(self, image_shape, base_white_noise_sigma = 0.5, random_seed = 0, complex_weight=False, device = 'cuda'):          
        assert len(image_shape) == 2 or len(image_shape) == 3, f'Input tensor should have 2 or 3 dimensions. The input tensor has {len(image_shape)} dimensions'
        assert image_shape[-1] == image_shape[-2], f'Input tensor should have the same height and width. The input tensor has height {image_shape[0]} and width {image_shape[1]}'        
        super().__init__()
        
        self.device = device        
        self.complex_weight = complex_weight
        self.num_channels = 1 if len(image_shape) == 2 else image_shape[0]
        self.H, self.W = image_shape[1], image_shape[2]
            
        if self.complex_weight:
            self.hermSymPara = ComplexHermitianSymmetricParamMultiChannel(image_shape[-2:], num_channels=self.num_channels).to(device)
        else:
            self.hermSymPara = HermitianSymmetricParamMultiChannel(image_shape[-2:], num_channels=self.num_channels).to(device)
            
        self.generator_torch = torch.Generator(device=device)
        self.generator_torch.manual_seed(random_seed)
        self.base_white_noise_sigma = base_white_noise_sigma
        
        
    # def compute_unit_spatial_var(self):
    #     return (self.hermSymPara().real ** 2 + self.hermSymPara().imag ** 2).sum((-1, -2)) / (self.H ** 2)
    
    
    def compute_unit_spatial_var(self):
        def compute_var():
            if self.complex_weight:
                return (self.hermSymPara().real ** 2 + self.hermSymPara().imag ** 2).sum((-1, -2)) / (self.H ** 2)
            else:
                return (self.hermSymPara() ** 2).sum((-1, -2)) / (self.H ** 2)
        
        if self.training:            
            return compute_var()
        else:
            if not hasattr(self, 'unit_spatial_var'):
                self.unit_spatial_var = compute_var()
            return self.unit_spatial_var

    def getNoise(self, x = None, shape = None, device = 'cuda'):        
        assert x is not None or shape is not None, f'Either the input tensor or the shape of the input tensor should be provided'       
        assert not (x is not None and shape is not None), f'Either the input tensor or the shape of the input tensor should be provided, not both'        
        
        if x is not None:
            assert len(x.shape) == 4, f'Input tensor should have 4 dimensions. The input tensor has {len(x.shape)} dimensions'
            batch_size, num_channels, H, W = x.shape
        else:
            batch_size, num_channels, H, W = shape
             
        assert num_channels == self.num_channels, f'Input tensor should have the same number of channels. The input tensor has {num_channels} channels while the model has {self.num_channels} channels'
        assert H == self.H, f'Input tensor should have the same height. The input tensor has height {H} while the model has height {self.H}'
        assert W == self.W, f'Input tensor should have the same width. The input tensor has width {W} while the model has width {self.W}'        
        
        white_noise = torch.randn( (batch_size, num_channels, H, W), generator = self.generator_torch, device = self.device)
        white_noise_base = torch.randn( (batch_size, num_channels, H, W), generator = self.generator_torch, device = self.device)
        
        white_noise_freq = torch.fft.fft2(white_noise, s = (H, W), norm = 'forward')
        white_noise_freq_mollified = white_noise_freq * self.hermSymPara().unsqueeze(0) 

        noise_mollified = torch.fft.ifft2(white_noise_freq_mollified, norm = 'forward') /  ( self.compute_unit_spatial_var()[None, : , None, None]  ** 0.5 )

        if self.training:
            return noise_mollified.real * (1 - self.base_white_noise_sigma ** 2) ** 0.5 + white_noise_base * self.base_white_noise_sigma
        else:
            return (noise_mollified.real * (1 - self.base_white_noise_sigma ** 2) ** 0.5 + white_noise_base * self.base_white_noise_sigma).detach() 
    
    
    def mollify(self, white_noise, white_noise_base = None, device = 'cuda'):        
        if type(white_noise) == np.ndarray:
            white_noise = torch.from_numpy(white_noise).to(device)
            return_numpy = True
        else:
            return_numpy = False
            
        assert len(white_noise.shape) == 4, f'Input tensor should have 4 dimensions. The input tensor has {len(white_noise.shape)} dimensions'
        
        _, _, H, W = white_noise.shape  
        
        if white_noise_base is None:
            white_noise_base = torch.randn_like(white_noise)
        else:
            assert len(white_noise_base.shape) == 4, f'Input tensor should have 4 dimensions. The input tensor has {len(white_noise_base.shape)} dimensions'
                
        white_noise_freq = torch.fft.fft2(white_noise, s = (H, W), norm = 'forward')
        white_noise_freq_mollified = white_noise_freq * self.hermSymPara().unsqueeze(0)
        noise_mollified = torch.fft.ifft2(white_noise_freq_mollified, norm = 'forward') /  ( self.compute_unit_spatial_var()[None, : , None, None]  ** 0.5 )        
        
        noise_to_return = noise_mollified.real * (1 - self.base_white_noise_sigma ** 2) ** 0.5 + white_noise_base * self.base_white_noise_sigma
        
        if return_numpy:
            return noise_to_return.detach().cpu().numpy()
        else:
            return noise_to_return
            
        # return noise_mollified.real * (1 - self.base_white_noise_sigma ** 2) ** 0.5 + white_noise_base * self.base_white_noise_sigma


