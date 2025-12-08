# src/masks/multi_green.py
import torch
import math
import torch.nn.functional as F
from multiprocessing import Value
from logging import getLogger

logger = getLogger()

class GreenMaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0), # Context scale
        pred_mask_scale=(0.6, 0.8), # Target scale (Total area of targets)
        sigma_1=0.5, # Green noise low freq sigma
        sigma_2=2.0, # Green noise high freq sigma
        min_keep=10,
        allow_overlap=False
    ):
        # TODO: super(MaskCollator, self).__init__() ?? 
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )  # total number of patches for x, y dimensions of the entire image
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock(): # Thread-safe atomic increment
            i.value += 1
            v = i.value
        return v

    def _generate_green_noise(self, generator):
        """
        Generates a Green Noise map (Band-pass filtered noise)
        Corresponds to Eq. 3 in ColorMAE paper: Ng = G_s1 * W - G_s2 * W
        """
        # 1. Generate White Noise (W)
        noise = torch.randn(1, 1, self.height, self.width, generator=generator)

        # 2. Define Gaussian Kernels
        # Kernel size should be large enough to cover the sigma interaction (e.g., 4*sigma)
        k_size = int(4 * self.sigma_2 + 1)
        if k_size % 2 == 0: k_size += 1
        
        # Create 2D Gaussian kernels
        # Note: In a production collator, these kernels should be cached to save CPU cycles
        x_cord = torch.arange(k_size).float() - k_size // 2
        x_grid = x_cord.repeat(k_size).view(k_size, k_size)
        y_grid = x_grid.t()
        xy_grid = torch.sqrt(x_grid**2 + y_grid**2)

        def get_kernel(sigma):
            kernel = torch.exp(-((xy_grid)**2) / (2 * sigma**2))
            kernel = kernel / torch.sum(kernel)
            return kernel.view(1, 1, k_size, k_size)

        k1 = get_kernel(self.sigma_1)
        k2 = get_kernel(self.sigma_2)

        # 3. Apply Filters (Band-pass)
        # We use padding to maintain spatial dimensions
        pad = k_size // 2
        
        # Weak blur (High pass component)
        blur1 = F.conv2d(noise, k1, padding=pad)
        # Strong blur (Low pass component)
        blur2 = F.conv2d(noise, k2, padding=pad)

        # Green Noise = Weak Blur - Strong Blur
        green_noise = blur1 - blur2
        return green_noise.squeeze()

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
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_pred = [] # Targets
        collated_masks_enc = []  # Contexts

        for _ in range(B):
            # 1. Generate Green Noise Map for this image
            # We use one map per image to ensure context/target come from same distribution logic
            # Or generate two independent ones? 
            # I-JEPA samples context and target independently. 
            # ColorMAE uses one map. Let's generate independent noise maps for diversity.
            
            # --- TARGET GENERATION ---
            noise_pred = self._generate_green_noise(g)
            
            # Select Target Mask (Top X% of Green Noise)
            # We pass None for acceptable_mask because Target is generated first
            pred_indices, pred_binary = self._threshold_mask(
                noise_pred, self.pred_mask_scale, g
            )
            
            # I-JEPA expects a list of targets (usually 4 blocks). 
            # In Green Noise, we generate one big "clutter" mask. 
            # We wrap it in a list to match the return signature expected by train.py
            masks_p = [pred_indices] 
            
            # --- CONTEXT GENERATION ---
            noise_enc = self._generate_green_noise(g)
            
            # Define Exclusion Zone (Inverse of Target) if overlap not allowed
            acceptable_mask = None
            if not self.allow_overlap:
                # Acceptable = 1 where Pred is 0
                acceptable_mask = 1 - pred_binary
            
            # Select Context Mask
            enc_indices, _ = self._threshold_mask(
                noise_enc, self.enc_mask_scale, g, acceptable_mask=acceptable_mask
            )
            masks_e = [enc_indices]

            collated_masks_pred.append(masks_p)
            collated_masks_enc.append(masks_e)

        # Collate (Pad to same length if necessary, though fixed scale usually results in fixed length)
        # For simplicity using standard collation, assuming fixed sizes:
        def collate_masks(mask_list):
            # Flatten list of lists -> batch of tensors
            # Structure: Batch -> Num_Masks -> Indices
            # Input is B x N_masks x Num_Indices
            # Output should be List[Tensor(B, Num_Indices)] of length N_masks
            
            # Since we only generate 1 complex mask per image for now:
            out = []
            num_masks = len(mask_list[0])
            for i in range(num_masks):
                batch_m = torch.stack([m[i] for m in mask_list])
                out.append(batch_m)
            return out[0] # Just return the tensor if it's a single block logic, 
                          # BUT train.py expects a list if you use multiple blocks.
                          # Adjust based on train.py loop.
        
        # Based on src/train.py:
        # masks_enc is passed to encoder. encoder expects [B, N] or list?
        # train.py line 205: z = encoder(imgs, masks_enc)
        # src/models/vision_transformer.py line 248: if masks is not None: x = apply_masks(x, masks)
        # src/masks/utils.py line 17: for m in masks: ...
        # It expects a LIST of tensors.
        
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        # The collate above creates [B, 1, N_keep]. We need to convert to list of [B, N_keep]
        # Because train.py handles list of masks.
        
        final_masks_enc = [collated_masks_enc[:, i, :] for i in range(collated_masks_enc.shape[1])]
        final_masks_pred = [collated_masks_pred[:, i, :] for i in range(collated_masks_pred.shape[1])]

        return collated_batch, final_masks_enc, final_masks_pred