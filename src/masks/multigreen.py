# src/masks/multi_green.py
import torch
import math
import numpy as np
import torch.nn.functional as F
from multiprocessing import Value
from logging import getLogger

logger = getLogger()

class MaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0), # Context scale
        pred_mask_scale=(0.6, 0.8), # Target scale (Total area of targets)
        data_path="/users/kutay.eroglu/datasets/green_noise_data_3072.npz",  # Path to green noise patterns
        min_keep=10,
        allow_overlap=False
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
        
        Args:
            generator: Random generator for reproducibility
        
        Returns:
            torch.Tensor: Green noise pattern of shape [height, width]
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

    def _threshold_mask(self, noise_map, scale_range, generator, acceptable_mask=None):
        """
        Selects top-k pixels from the noise map to form the mask.
        """
        num_patches = self.height * self.width
        
        # Sample a target ratio (e.g., 0.6 to 0.8)
        min_s, max_s = scale_range
        rand_val = torch.rand(1, generator=generator).item()
        target_scale = min_s + rand_val * (max_s - min_s)
        
        num_keep = int(num_patches * target_scale)
        
        # If we have an exclusion zone (acceptable_mask), we force those pixels to -inf
        # so they are never selected
        if acceptable_mask is not None:
            # acceptable_mask is 1 where valid, 0 where invalid (overlap)
            # We want to forbid selecting where acceptable_mask == 0
            # noise_map[acceptable_mask == 0] = -float('inf')
            
            # NOTE: I-JEPA logic is slightly different: 
            # It generates the mask, THEN removes overlap. 
            # If we want strict I-JEPA adherence:
            pass

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

    def __call__(self, batch):
        """
        Process a batch of images and generate green noise-based masks.
        
        For each image:
        1. Extract a green noise pattern window (height x width)
        2. Use it to select target patches (top X% by noise value)
        3. Use it to select context patches (top Y% by noise value, avoiding targets if no overlap)
        """
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        # Use a seed for reproducibility
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_pred = []  # Target masks (what to predict)
        collated_masks_enc = []    # Context masks (what encoder sees)

        for img_idx in range(B):
            # ============================================================
            # STEP 1: Get a green noise pattern for this image
            # ============================================================
            # Extract a [height, width] window from loaded patterns
            # This gives us noise values at each patch position
            green_noise = self._get_green_noise_window(g)  # Shape: [height, width]
            
            # ============================================================
            # STEP 2: Create TARGET mask (what the model should predict)
            # ============================================================
            # Select top X% of patches (by noise value) as targets
            # pred_mask_scale = (0.6, 0.8) means keep 60-80% of patches
            pred_indices, pred_binary = self._threshold_mask(
                green_noise,           # Use noise values to decide
                self.pred_mask_scale,  # Keep 60-80% of patches
                g,
                acceptable_mask=None   # No restrictions for targets
            )
            # pred_indices: which patch indices to keep (e.g., [5, 12, 23, ...])
            # pred_binary: binary mask [height, width] where 1 = keep, 0 = mask
            
            masks_p = [pred_indices]  # Wrap in list (I-JEPA expects list of masks)
            
            # ============================================================
            # STEP 3: Create CONTEXT mask (what encoder sees)
            # ============================================================
            # Get a DIFFERENT green noise pattern for context (for diversity)
            green_noise_enc = self._get_green_noise_window(g)  # Different pattern
            
            # If overlap not allowed, exclude target patches from context
            acceptable_mask = None
            if not self.allow_overlap:
                # acceptable_mask = 1 where we CAN select, 0 where we CANNOT
                # We cannot select patches that are already targets
                acceptable_mask = 1 - pred_binary  # Inverse of target mask
            
            # Select top Y% of patches (by noise value) as context
            # enc_mask_scale = (0.85, 1.0) means keep 85-100% of patches
            enc_indices, _ = self._threshold_mask(
                green_noise_enc,       # Use noise values to decide
                self.enc_mask_scale,   # Keep 85-100% of patches
                g,
                acceptable_mask=acceptable_mask  # Respect overlap constraint
            )
            
            masks_e = [enc_indices]  # Wrap in list
            
            collated_masks_pred.append(masks_p)
            collated_masks_enc.append(masks_e)

        # ============================================================
        # STEP 4: Collate masks for the batch
        # ============================================================
        # Convert from list of lists to batch tensors
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        # Shape after collate: [B, 1, N_keep] where N_keep = number of kept patches
        # train.py expects a LIST of tensors, each of shape [B, N_keep]
        final_masks_enc = [collated_masks_enc[:, i, :] for i in range(collated_masks_enc.shape[1])]
        final_masks_pred = [collated_masks_pred[:, i, :] for i in range(collated_masks_pred.shape[1])]

        return collated_batch, final_masks_enc, final_masks_pred