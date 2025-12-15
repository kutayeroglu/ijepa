import math
from logging import getLogger
from multiprocessing import Value

import numpy as np
import torch
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()

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
        
        # Load green noise patterns from file
        self._load_green_noise_patterns(data_path)

    def step(self):
        i = self._itr_counter
        with i.get_lock(): # Thread-safe atomic increment
            i.value += 1
            v = i.value
        return v

    def _sample_mask_scale(self, generator, scale):
        """Sample a mask scale value from the given range."""
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        return mask_scale

    def _load_green_noise_patterns(self, data_path):
        """Load pre-generated green noise patterns from npz file."""
        try:
            image_tensor = np.load(data_path)
            image_tensor = torch.from_numpy(image_tensor[image_tensor.files[0]])
            print(f"=========> Loading Green Noise Patterns: {data_path} <=========")
        except Exception as e:
            raise Exception(f"Green Noise patterns not found at {data_path}. Error: {e}")
        
        self.green_noise_tensor = image_tensor  # Shape: [L, M, N]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.green_noise_tensor = self.green_noise_tensor.to(self.device)

    def _get_green_noise_window(self, generator):
        """
        Extract a [height, width] green noise pattern window from loaded patterns.
        """
        L, M, N = self.green_noise_tensor.shape  # [num_patterns, pattern_h, pattern_w]
        
        # Step 1: Randomly select one pattern from the L available patterns
        pattern_idx = torch.randint(0, L, (1,), generator=generator).item()
        pattern = self.green_noise_tensor[pattern_idx]  # Shape: [M, N]
        
        # Step 2: Extract a [height, width] window from the pattern
        if M >= self.height and N >= self.width:
            # Pattern is larger - randomly crop a window
            top = torch.randint(0, M - self.height + 1, (1,), generator=generator).item()
            left = torch.randint(0, N - self.width + 1, (1,), generator=generator).item()
            window = pattern[top:top+self.height, left:left+self.width]
        elif M == self.height and N == self.width:
            # Pattern matches exactly - use it directly
            window = pattern
        else:
            # Pattern is smaller - interpolate (shouldn't happen with proper patterns)
            window = F.interpolate(
                pattern.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Step 3: Optional random flips for additional variety
        if torch.rand(1, generator=generator).item() < 0.5:
            window = torch.flip(window, dims=[1])  # horizontal flip
        if torch.rand(1, generator=generator).item() < 0.5:
            window = torch.flip(window, dims=[0])  # vertical flip
        
        return window  # Shape: [height, width]

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
             # Set noise in 'forbidden' regions to lowest possible value
             flat_noise = flat_noise.masked_fill(acceptable_mask.flatten() == 0, -1e9)

        _, indices = torch.sort(flat_noise, descending=True)
        keep_indices = indices[:num_keep]
        
        # Create binary mask for overlap checking
        binary_mask = torch.zeros_like(noise_map)
        binary_mask.view(-1)[keep_indices] = 1
        
        return keep_indices, binary_mask

    def _sample_noise_mask(self, scale_value, generator, acceptable_regions=None):
        """
        Sample a mask from green noise pattern given a scale value.
        Similar to _sample_block_mask but uses green noise instead of rectangular blocks.
        """
        green_noise = self._get_green_noise_window(generator)

        def constrain_mask(mask, tries=0):
            """
            Helper to restrict given mask to a set of acceptable regions.
            
            Args:
                mask: 2D binary mask (1 = in mask, 0 = not in mask)
                tries: Number of constraint relaxations (0 = enforce all, 1 = ignore one, etc.)
            
            Process:
                - acceptable_regions is a list of 2D binary masks (1 = acceptable, 0 = not acceptable)
                - Element-wise multiplication: mask *= region zeros out pixels where region=0
                - This "crops" the mask to only overlap with acceptable regions
                - As tries increases, fewer regions are enforced (gradual constraint relaxation)
            """
            if acceptable_regions is None:
                return mask
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask = mask * acceptable_regions[k]  # Element-wise: 1*1=1 (keep), 1*0=0 (remove)
            return mask

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # Get a green noise pattern window
            green_noise = self._get_green_noise_window(generator)
            
            # Convert acceptable_regions to acceptable_mask format for _threshold_mask
            acceptable_mask = None
            if acceptable_regions is not None:
                # Combine acceptable regions (with tries-based reduction)
                if len(acceptable_regions) > 0:
                    acceptable_mask = torch.ones(
                        (self.height, self.width), 
                        dtype=torch.float32, 
                        device=green_noise.device
                    )
                    N = max(int(len(acceptable_regions) - tries), 0)
                    for k in range(N):
                        # acceptable_regions[k] is a binary mask where 1 = acceptable
                        acceptable_mask = acceptable_mask * acceptable_regions[k].float()
            
            # Create mask using threshold
            mask_indices, binary_mask = self._threshold_mask(
                noise_map=green_noise,
                scale=scale_value,
                acceptable_mask=acceptable_mask
            )
            
            # Check if mask is valid (has enough patches)
            valid_mask = len(mask_indices) >= self.min_keep
            
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        
        # Create complement mask (1 where mask is 0, 0 where mask is 1)
        mask_complement = 1 - binary_mask
        
        return mask_indices, mask_complement



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
        min_keep_pred = self.height * self.width # maximum num of patches -> safe upper bound
        min_keep_enc = self.height * self.width # maximum num of patches -> safe upper bound

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_noise_mask(p_size, g)
                masks_p.append(mask)  # target block mask
                masks_C.append(mask_C)  # target block complement mask
                min_keep_pred = min(min_keep_pred, len(mask)) # update minimum number of patches to keep for target block
            collated_masks_pred.append(masks_p)
                
            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f"Encountered exception in mask-generator {e}")

        
        
        
        
        
        green_noise = self._get_green_noise_window(g) 




        pred_indices, pred_binary = self._threshold_mask(
            green_noise,            
            p_size,   # Use the pre-sampled scale instead of self.pred_mask_scale
            acceptable_mask=None    
        )

        enc_indices, _ = self._threshold_mask(
            green_noise_enc,       
            e_size,   # Use the pre-sampled scale instead of self.enc_mask_scale
            acceptable_mask=acceptable_mask 
        )







 

        for img_idx in range(B):
            # ============================================================
            # STEP 1: Get a green noise pattern for this image
            # ============================================================
            green_noise = self._get_green_noise_window(g) 
            
            # ============================================================
            # STEP 2: Create TARGET mask
            # ============================================================
            pred_indices, pred_binary = self._threshold_mask(
                green_noise,            
                self.pred_mask_scale,   
                acceptable_mask=None    
            )
            
            masks_p = [pred_indices]  # Wrap in list
            
            # ============================================================
            # STEP 3: Create CONTEXT mask
            # ============================================================
            green_noise_enc = self._get_green_noise_window(g) 
            
            acceptable_mask = None
            if not self.allow_overlap:
                acceptable_mask = 1 - pred_binary 
            
            enc_indices, _ = self._threshold_mask(
                green_noise_enc,       
                self.enc_mask_scale,   
                acceptable_mask=acceptable_mask 
            )
            
            masks_e = [enc_indices]  # Wrap in list
            
            collated_masks_pred.append(masks_p)
            collated_masks_enc.append(masks_e)

        # ============================================================
        # STEP 4: Synchronize Lengths (The Fix)
        # ============================================================
        # Calculate the minimum length found in this batch for both sets of masks
        # collated_masks_pred is a list of lists of tensors: [[tensor(140)], [tensor(145)], ...]
        min_keep_pred = min([len(m[0]) for m in collated_masks_pred])
        min_keep_enc = min([len(m[0]) for m in collated_masks_enc])
        
        # Truncate all masks to the minimum length
        # We slice indices[:min_keep] to ensure they are all the same size
        collated_masks_pred = [[m[0][:min_keep_pred]] for m in collated_masks_pred]
        collated_masks_enc = [[m[0][:min_keep_enc]] for m in collated_masks_enc]

        # ============================================================
        # STEP 5: Collate
        # ============================================================
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        # Shape after collate: [B, 1, N_keep]
        # Flatten the list dimension to match I-JEPA training loop expectations
        final_masks_enc = [collated_masks_enc[:, i, :] for i in range(collated_masks_enc.shape[1])]
        final_masks_pred = [collated_masks_pred[:, i, :] for i in range(collated_masks_pred.shape[1])]

        return collated_batch, final_masks_enc, final_masks_pred