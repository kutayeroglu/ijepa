# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


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
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():  # Thread-safe atomic increment
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
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

        # e.g., if p_size = (8, 12), the block is 8 patches tall and 12 patches wide
        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size  # block height and width (expressed in number of patches)

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
            top = torch.randint(0, self.height - h, (1,))  # maximum valid top position
            left = torch.randint(0, self.width - w, (1,))  # maximum valid left position
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1  # mark masked region as 1
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
        # Create complement mask (2D, not flattened)
        # mask_complement stays 2D because it's used for spatial constraints:
        # - It becomes part of acceptable_regions
        # - Used in element-wise multiplication with 2D masks in constrain_mask
        # - The spatial structure (height × width) is needed for constraint logic
        # NOTE: mask is 1D indices (for indexing), but mask_complement is 2D (for spatial ops)
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
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

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(  # block size for target block [h,w (in number of patches)]
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(  # block size for context block [h,w (in number of patches)]
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width # maximum num of patches -> safe upper bound
        min_keep_enc = self.height * self.width # maximum num of patches -> safe upper bound
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
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

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask)) # update minimum number of patches to keep for context block
            collated_masks_enc.append(masks_e)

        # ensure all masks are the same size (truncate to minimum)
        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
