# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import numpy as np
import torch
from torchvision import transforms

_GLOBAL_SEED = 0
logger = getLogger()


class NormalizeBySliceMax:
    """Normalize each slice of the image tensor by its maximum."""

    def __init__(self):
        pass

    def __call__(self, img):
        # Assuming img is a PyTorch tensor with shape [L, W, W]
        max_values = img.max(dim=-1).values.max(dim=-1).values
        max_values = max_values.unsqueeze(1).unsqueeze(2)
        return img / max_values

    def __repr__(self):
        return self.__class__.__name__


class MaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8), # total fraction of patches to mask
        pred_mask_scale=(0.2, 0.8), # total fraction of patches to mask
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        debug_log=False,
        color_noise_path="noise_colors/green/green_noise_data_3072.npz",
        color_mask_ratio=0.15, # ratio of patches to drop from initially selected blocks
        enc_drop_order="lowest",  # "lowest" | "highest" - which noise-value patches to drop for context
        pred_drop_order="lowest",  # "lowest" | "highest" - which noise-value patches to drop for target
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )  # total number of patches for x, y dimensions of the entire image
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = (
            allow_overlap  # whether to allow overlap b/w enc and pred masks
        )
        self.debug_log = debug_log
        self._debug_logged_batches = 0  # only log full detail for first batch(s)
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes
        if self.debug_log:
            logger.info(
                "[multiblock] __init__: input_size=%s patch_size=%s height=%s width=%s "
                "enc_mask_scale=%s pred_mask_scale=%s aspect_ratio=%s nenc=%s npred=%s "
                "min_keep=%s allow_overlap=%s",
                input_size, patch_size, self.height, self.width,
                enc_mask_scale, pred_mask_scale, aspect_ratio, nenc, npred,
                min_keep, allow_overlap,
            )

        # -- Color Noise Initialization
        self.color_mask_ratio = color_mask_ratio
        for name, val in [("enc_drop_order", enc_drop_order), ("pred_drop_order", pred_drop_order)]:
            if val not in ("lowest", "highest"):
                raise ValueError(f"{name} must be 'lowest' or 'highest', got {val!r}")
        self.enc_drop_order = enc_drop_order
        self.pred_drop_order = pred_drop_order
        self.trans_sequence = transforms.Compose([
            transforms.RandomCrop(self.height), # Crop to [self.height, self.width] which is [14, 14] for ViT/14
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            NormalizeBySliceMax()
        ])
        self._load_color_pattern(color_noise_path)

    def _load_color_pattern(self, data_path):
        try:
            image_tensor = np.load(data_path)
            image_tensor = torch.from_numpy(image_tensor[image_tensor.files[0]]) 
            if "green" in data_path:
                logger.info(f"=========> Loading Green Noise Patterns: {data_path} <=========")
            elif "blue" in data_path:
                logger.info(f"=========> Loading Blue Noise Patterns: {data_path} <=========")
            elif "purple" in data_path:
                logger.info(f"=========> Loading Purple Noise Patterns: {data_path} <=========")
            elif "red" in data_path:
                logger.info(f"=========> Loading Red Noise Patterns: {data_path} <=========")
            elif "white" in data_path:
                logger.info(f"=========> Loading White Noise Patterns: {data_path} <=========")

        except Exception as e:
            raise Exception(f"Color Noise patterns not found at {data_path}. Error: {e}")
            
        self.image_tensor = image_tensor.float()

    def _extract_noise_windows(self, B: int) -> torch.Tensor:
        """
        Produce B augmented noise grids from the stored noise pattern collection.

        self.image_tensor has shape [L, M, N] where L is the number of
        stored noise patterns. Each call to self.trans_sequence applies a
        random transformation (crop, flip, normalize) to ALL L patterns at
        once, yielding [L, height, width].

        To cover a batch of size B:
          - full_iterations = B // L  full passes, each transforming all L
            patterns with a fresh random augmentation.
          - If B % L > 0, one additional pass transforms all L patterns and
            only the first `residual` grids are kept.

        Returns:
            Tensor of shape [B, height, width].
        """
        L, M, N = self.image_tensor.shape  
        windows = []
        full_iterations = B // L
        residual = B % L

        for _ in range(full_iterations):
            w_tensor = self.trans_sequence(self.image_tensor) 
            windows.append(w_tensor) 
            
        if residual > 0:
            w_tensor = self.trans_sequence(self.image_tensor)[:residual]
            windows.append(w_tensor)

        return torch.concatenate(windows, dim=0)

    def step(self):
        i = self._itr_counter
        with i.get_lock():  # Thread-safe atomic increment
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale, label=None, log_detail=False):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        if log_detail and label is not None:
            logger.info(
                "[multiblock] _sample_block_size(%s): scale=%s mask_scale=%s max_keep=%s "
                "aspect_ratio_scale=%s aspect_ratio=%s (h,w)=(%s,%s)",
                label, scale, mask_scale, max_keep, aspect_ratio_scale, aspect_ratio, h, w,
            )
        # e.g., if p_size = (8, 12), the block is 8 patches tall and 12 patches wide
        return (h, w)

    def _get_quadrant_bounds(self) -> list[tuple[int, int, int, int]]:
        """
        Return quadrant bounds on the patch grid as
        (top_min, top_max, left_min, left_max) with exclusive upper bounds.
        """
        h_mid = self.height // 2
        w_mid = self.width // 2
        return [
            (0, h_mid, 0, w_mid),
            (0, h_mid, w_mid, self.width),
            (h_mid, self.height, 0, w_mid),
            (h_mid, self.height, w_mid, self.width),
        ]

    def _get_top_left_bounds(
        self,
        b_size: tuple[int, int],
        region_bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """
        Return valid top-left sampling bounds for a block, optionally
        intersected with a region on the patch grid.
        """
        h, w = b_size
        top_min, top_max = 0, self.height - h
        left_min, left_max = 0, self.width - w

        if region_bounds is not None:
            region_top_min, region_top_max, region_left_min, region_left_max = region_bounds
            top_min = max(top_min, region_top_min)
            top_max = min(top_max, region_top_max)
            left_min = max(left_min, region_left_min)
            left_max = min(left_max, region_left_max)

        if top_max <= top_min or left_max <= left_min:
            return None
        return (top_min, top_max, left_min, left_max)

    def _sample_noisy_block_mask(
        self, 
        b_size: tuple[int, int], 
        noise_grid: torch.Tensor, # shape: [self.height, self.width]
        acceptable_regions: list[torch.Tensor] | None = None, 
        log_detail: bool = False,
        drop_order: str = "lowest",
        top_left_bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = b_size  # block height and width (expressed in number of patches)
        sampling_bounds = top_left_bounds or self._get_top_left_bounds(b_size)
        if sampling_bounds is None:
            raise ValueError(f"Could not compute valid sampling bounds for block size {b_size}")
        top_min, top_max, left_min, left_max = sampling_bounds

        def constrain_mask(mask, tries=0):
            """
            Helper to restrict given mask to a set of acceptable regions.
            
            Args:
                mask: 2D binary mask (1 = in block, 0 = not in block)
                tries: Number of constraint relaxations (0 = enforce all, 1 = ignore one, etc.)
            
            Process:
                - acceptable_regions is a list of 2D binary masks (1 = acceptable, 0 = not acceptable)
                - Element-wise multiplication: mask *= region zeros out pixels where region=0
                - This "crops" the block to only overlap with acceptable regions
                - As tries increases, fewer regions are enforced (gradual constraint relaxation)
            """            
            if acceptable_regions is None:
                return
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]  # Element-wise: 1*1=1 (keep), 1*0=0 (remove)
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner -> this ensures the block is within the image
            top = torch.randint(top_min, top_max, (1,))
            left = torch.randint(left_min, left_max, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1  # mark masked region as 1
            
            # --- COLOR NOISE THRESHOLDING
            # Drop color_mask_ratio of patches from the block, keeping those
            # with the highest noise values (spatially-structured dropout).
            if self.color_mask_ratio > 0.0:
                # 1. Get the current active patches in the box
                box_indices_2d = torch.nonzero(mask)
                total_in_box = box_indices_2d.shape[0]
                
                if total_in_box > 0:
                    # 2. Extract specifically the noise values for those patches
                    box_noise_values = noise_grid[box_indices_2d[:, 0], box_indices_2d[:, 1]]
                    
                    # 3. Sort those noise values (descending=keep highest/drop lowest, ascending=keep lowest/drop highest)
                    descending = (drop_order == "lowest")
                    ids_shuffle = torch.argsort(box_noise_values, descending=descending)
                    
                    # 4. Calculate how many to keep
                    len_keep = int(total_in_box * (1 - self.color_mask_ratio))
                    
                    # 5. Get the indices (relative to the box elements) of the ones we drop
                    ids_drop = ids_shuffle[len_keep:]
                    
                    # 6. Map those dropped localized indices back to their 2D image coordinates
                    drop_coords_2d = box_indices_2d[ids_drop]
                    
                    # 7. Set those coordinates back to 0 in the 2D mask!
                    mask[drop_coords_2d[:, 0], drop_coords_2d[:, 1]] = 0
            # --- 
            
            # Capture 2D mask after noise thresholding for accurate complement
            mask_2d = mask.clone()

            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten()) # Converts 2D binary mask to indices
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        # Complement from the actual noise-thresholded mask (2D) rather than
        # the full rectangle. This way acceptable_regions only excludes patches
        # that are truly prediction targets, not ones removed by noise.
        mask_complement = 1 - mask_2d
        # --
        if log_detail:
            top_val = top.item() if top.dim() > 0 else int(top)
            left_val = left.item() if left.dim() > 0 else int(left)
            logger.info(
                "[multiblock] _sample_noisy_block_mask: b_size=%s top=%s left=%s len(mask)=%s tries=%s",
                b_size, top_val, left_val, len(mask), tries,
            )
        return mask, mask_complement

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        B = len(batch)
        log_detail = self.debug_log and (getattr(self, "_debug_logged_batches", 0) == 0)
        if self.debug_log:
            logger.info("[multiblock] __call__: B=%s", B)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(  # block size for target block [h,w (in number of patches)]
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
            label="pred",
            log_detail=log_detail,
        )
        e_size = self._sample_block_size(  # block size for context block [h,w (in number of patches)]
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
            label="enc",
            log_detail=log_detail,
        )

        # Generate continuous color noise field for a batch of size B
        # batch_noise_grids shape: [B, self.height, self.width]
        batch_noise_grids = self._extract_noise_windows(B)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width # maximum num of patches -> safe upper bound
        min_keep_enc = self.height * self.width # maximum num of patches -> safe upper bound
        for i in range(B):
            # The current image's global noise field
            noise_grid = batch_noise_grids[i]
            
            masks_p, masks_C = [], []
            quadrant_bounds = self._get_quadrant_bounds()
            pred_quadrant_perm = torch.randperm(len(quadrant_bounds)) if self.npred <= len(quadrant_bounds) else None
            for pred_idx in range(self.npred):
                top_left_bounds = None
                if pred_quadrant_perm is not None:
                    quadrant_idx = pred_quadrant_perm[pred_idx].item()
                    top_left_bounds = self._get_top_left_bounds(
                        b_size=p_size, 
                        region_bounds=quadrant_bounds[quadrant_idx],
                    )
                mask, mask_C = self._sample_noisy_block_mask(
                    b_size=p_size,
                    noise_grid=noise_grid,
                    log_detail=log_detail,
                    drop_order=self.pred_drop_order,
                    top_left_bounds=top_left_bounds,
                )
                masks_p.append(mask)  # target block mask
                masks_C.append(mask_C)  # target block complement mask
                min_keep_pred = min(min_keep_pred, len(mask)) # track min size so all target masks can be truncated to the same length for batch collation
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f"Encountered exception in mask-generator {e}")

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_noisy_block_mask(
                    b_size=e_size,
                    noise_grid=noise_grid,
                    acceptable_regions=acceptable_regions,
                    log_detail=log_detail,
                    drop_order=self.enc_drop_order
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask)) # track min size so all context masks can be truncated to the same length for batch collation
            collated_masks_enc.append(masks_e)

        # =========================================================================
        # Data Structure Explanation:
        # Before truncation, `collated_masks_pred` and `collated_masks_enc` are 
        # a list of lists of 1D torch.Tensors:
        #   - Outer list: Length B (batch size)
        #   - Inner list: Length npred/nenc (number of mask blocks per image)
        #   - Innermost item: 1D torch.Tensor of patch indices
        # 
        # We must truncate all 1D tensors to the exact same minimum length 
        # (`min_keep_pred` / `min_keep_enc`) so that PyTorch can stack them.
        # =========================================================================
        
        # ensure all masks are the same size (truncate to minimum)
        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        
        # After default_collate, it "zips" the batch items together.
        # It becomes a list of 2D torch.Tensors:
        #   - List length: npred/nenc (number of mask blocks)
        #   - Tensor shape: [B, min_keep_pred] (each block batched across images)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        if self.debug_log:
            logger.info(
                "[multiblock] __call__: min_keep_pred=%s min_keep_enc=%s",
                min_keep_pred, min_keep_enc,
            )
        if log_detail:
            _k = 20
            logger.info(
                "[multiblock] collated_masks_pred length=%s shape[0]=%s sample [0][0,:%s]=%s",
                len(collated_masks_pred),
                tuple(collated_masks_pred[0].shape),
                _k,
                collated_masks_pred[0][0, :_k].tolist(),
            )
            logger.info(
                "[multiblock] collated_masks_enc length=%s shape[0]=%s sample [0][0,:%s]=%s",
                len(collated_masks_enc),
                tuple(collated_masks_enc[0].shape),
                _k,
                collated_masks_enc[0][0, :_k].tolist(),
            )
        if self.debug_log:
            self._debug_logged_batches += 1

        return collated_batch, collated_masks_enc, collated_masks_pred
