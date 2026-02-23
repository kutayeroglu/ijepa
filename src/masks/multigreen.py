from logging import getLogger
from multiprocessing import Value

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

_GLOBAL_SEED = 0
logger = getLogger()

class NormalizeBySliceMax:
    """Normalize each slice of the image tensor by its maximum."""

    def __init__(self):
        pass

    def __call__(self, img):
        # Assuming img is a PyTorch tensor with shape [L, W, W]
        # Calculate max per spatial dimensions
        max_values = img.max(dim=-1).values.max(dim=-1).values
        # Only unsqueeze if it's a 3D tensor (L, H, W)
        if img.dim() == 3:
            max_values = max_values.unsqueeze(1).unsqueeze(2)
        elif img.dim() == 2:
            pass # max_values is a scalar
        return img / max_values

    def __repr__(self):
        return self.__class__.__name__

class MaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0), # Context scale
        pred_mask_scale=(0.6, 0.8), # Target scale (Total area of targets)
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        data_path="/users/kutay.eroglu/datasets/green_noise_data_3072.npz",  # Path to green noise patterns
    ):
        super(MaskCollator, self).__init__() 
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )  # total number of patches for x, y dimensions of the entire image
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)
        
        self._load_green_noise_patterns(data_path)
        
        # Initialize the transforms sequence similar to colorMAE
        self.trans_sequence = transforms.Compose([
            transforms.RandomCrop((self.height, self.width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            NormalizeBySliceMax()
        ])

    def _load_green_noise_patterns(self, data_path):
        """Load pre-generated green noise patterns from npz file."""
        try:
            print(f"=========> Loading Green Noise Patterns: {data_path} <=========")
            image_tensor = np.load(data_path)
            image_tensor = torch.from_numpy(image_tensor[image_tensor.files[0]])
        except Exception as e:
            raise Exception(f"Green Noise patterns not found at {data_path}. Error: {e}")
        
        self.green_noise_tensor = image_tensor  # NOTE: Shape: [L, M, N], see _get_green_noise_window
        self.green_noise_tensor = self.green_noise_tensor.float()
    
    def step(self):
        i = self._itr_counter
        with i.get_lock(): # Thread-safe atomic increment
            i.value += 1
            v = i.value
        return v

    def _sample_mask_scale(self, generator, scale) -> float:
        """Sample a mask scale value from the given range."""
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        return mask_scale

    def _get_green_noise_window(self, generator=None):
        """
        Extract a [height, width] green noise pattern window from loaded patterns using transforms.
        """
        L = self.green_noise_tensor.shape[0]  # [num_patterns, pattern_h, pattern_w]
        
        # Randomly select one pattern from the L available patterns
        # We handle generator explicitly because torch.randint handles RNG state directly while transforms relies on global/built-in RNG state if not managed
        if generator is not None:
            pattern_idx = torch.randint(0, L, (1,), generator=generator).item()
        else:
            pattern_idx = torch.randint(0, L, (1,)).item()
            
        pattern = self.green_noise_tensor[pattern_idx]  # Shape: [M, N]
        
        # Add a dummy batch dimension because RandomCrop expects at least 3D or specific Image types in older torchvision
        # but modern torchvision can handle 2D. We unsqueeze to [1, M, N] to be safe.
        pattern = pattern.unsqueeze(0)
        
        # Apply the transforms sequence (RandomCrop, Flips, Normalize)
        window = self.trans_sequence(pattern)
        
        # Remove the dummy channel dimension to return [height, width]
        return window.squeeze(0)

    def _threshold_mask(self, noise_map, scale, acceptable_mask=None):
        """
        Selects top-k pixels from the noise map to form the mask.
        """
        num_patches = self.height * self.width
        num_keep = int(num_patches * scale)
        
        # Sort and pick top k
        flat_noise = noise_map.flatten()
        
        # If we need to respect overlap constraints *during* generation:
        if acceptable_mask is not None:
            # Check if mask is an inverse mask (0=allowed, 1=forbidden)
            flat_noise = flat_noise.masked_fill(acceptable_mask.flatten() == 1, float('-inf'))

        # Add small random noise to break ties if multiple patches have exact same value or -inf
        flat_noise = flat_noise + torch.rand_like(flat_noise) * 1e-6

        _, indices = torch.sort(flat_noise, descending=True)
        keep_indices = indices[:num_keep]
        
        # Create binary mask for overlap checking (1 means forbidden/used)
        binary_mask = torch.zeros_like(noise_map)
        binary_mask.view(-1)[keep_indices] = 1
        
        return keep_indices, binary_mask

    def __call__(self, batch):
        """
        Process a batch of images and generate green noise-based masks.
        Ensures consistent tensor sizes for collation by truncating to minimum length.
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_mask_scale(  # sampled scale for target block [h,w (in number of patches)]
            generator=g,
            scale=self.pred_mask_scale,
        )
        e_size = self._sample_mask_scale(  # sampled scale for context block [h,w (in number of patches)]
            generator=g,
            scale=self.enc_mask_scale,
        )

        collated_masks_pred, collated_masks_enc = [], []  

        for img_idx in range(B):
            masks_p = []
            masks_C_list = []  # track all predictor binary masks for this image
            
            # 1. Generate Predictor (Target) Masks
            for _ in range(self.npred):
                green_noise = self._get_green_noise_window(g) 
                
                # Create TARGET mask
                pred_indices, pred_binary = self._threshold_mask(
                    green_noise,            
                    p_size,   
                    acceptable_mask=None    
                )
                masks_p.append(pred_indices)
                masks_C_list.append(pred_binary)
                
            collated_masks_pred.append(masks_p)
            
            # Combine all predictor constraints if no overlap allowed
            acceptable_mask = None
            if not self.allow_overlap and len(masks_C_list) > 0:
                # 1 means forbidden in our updated logic
                acceptable_mask = torch.clamp(sum(masks_C_list), 0, 1)

            # 2. Generate Encoder (Context) Masks
            masks_e = []
            for _ in range(self.nenc):
                green_noise_enc = self._get_green_noise_window(g) 
                
                enc_indices, _ = self._threshold_mask(
                    green_noise_enc,       
                    e_size,   
                    acceptable_mask=acceptable_mask 
                )
                masks_e.append(enc_indices)
                
            collated_masks_enc.append(masks_e)

        # Calculate the minimum length found in this batch for both sets of masks
        # collated_masks_pred shape: [B, npred, tensor(len)]
        min_keep_pred = min([len(m) for img_masks in collated_masks_pred for m in img_masks])
        min_keep_enc = min([len(m) for img_masks in collated_masks_enc for m in img_masks])
        
        # Truncate all masks to the minimum length
        collated_masks_pred = [[m[:min_keep_pred] for m in img_masks] for img_masks in collated_masks_pred]
        collated_masks_enc = [[m[:min_keep_enc] for m in img_masks] for img_masks in collated_masks_enc]

        # Collate list of lists of tensors
        # Returns list of shape npred, each item a tensor [B, min_keep]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred