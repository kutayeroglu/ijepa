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
    """BEiT-style blockwise masks for I-JEPA (union of multiple rectangles per mask)."""

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        debug_log=False,
        min_block_patches=4,
        placement_attempts=10,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.debug_log = debug_log
        self.min_block_patches = min_block_patches
        self.placement_attempts = placement_attempts
        self._debug_logged_batches = 0
        self._itr_counter = Value("i", -1)
        if self.debug_log:
            logger.info(
                "[blockwise] __init__: input_size=%s patch_size=%s height=%s width=%s "
                "enc_mask_scale=%s pred_mask_scale=%s aspect_ratio=%s nenc=%s npred=%s "
                "min_keep=%s allow_overlap=%s min_block_patches=%s placement_attempts=%s",
                input_size, patch_size, self.height, self.width,
                enc_mask_scale, pred_mask_scale, aspect_ratio, nenc, npred,
                min_keep, allow_overlap, min_block_patches, placement_attempts,
            )

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_budget(self, generator, scale, label=None, log_detail=False):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        num_patches = int(self.height * self.width * mask_scale)
        if log_detail and label is not None:
            logger.info(
                "[blockwise] _sample_mask_budget(%s): scale=%s mask_scale=%s num_patches=%s",
                label, scale, mask_scale, num_patches,
            )
        return num_patches

    def _place_rectangle(self, mask_2d, max_new_patches, aspect_ratio_range):
        """Place one axis-aligned rectangle on mask_2d (BEiT overlap rule). Returns new patch count."""
        min_ar, max_ar = aspect_ratio_range
        log_min_ar = math.log(min_ar)
        log_max_ar = math.log(max_ar)
        delta = 0
        for _ in range(self.placement_attempts):
            lo = max(self.min_block_patches, 1)
            hi = max(max_new_patches, lo)
            target_area = lo + torch.rand(1).item() * (hi - lo)
            aspect = math.exp(log_min_ar + torch.rand(1).item() * (log_max_ar - log_min_ar))
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if h < 1 or w < 1:
                continue
            if h >= self.height or w >= self.width:
                continue
            top = torch.randint(0, self.height - h, (1,)).item()
            left = torch.randint(0, self.width - w, (1,)).item()
            patch = mask_2d[top : top + h, left : left + w]
            unmasked = patch == 0
            new_pixels = int(unmasked.sum().item())
            if 0 < new_pixels <= max_new_patches:
                patch[unmasked] = 1
                delta = new_pixels
                break
        return delta

    def _compose_blockwise_mask(self, num_patches, aspect_ratio_range):
        """Iteratively place rectangles until num_patches is reached or placement stalls."""
        mask_2d = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask_count = 0
        while mask_count < num_patches:
            max_new = num_patches - mask_count
            delta = self._place_rectangle(mask_2d, max_new, aspect_ratio_range)
            if delta == 0:
                break
            mask_count += delta
        return mask_2d

    def _build_blockwise_mask(
        self,
        num_patches,
        aspect_ratio_range,
        acceptable_regions=None,
        log_detail=False,
    ):
        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            mask_2d = self._compose_blockwise_mask(num_patches, aspect_ratio_range)
            if acceptable_regions is not None:
                constrain_mask(mask_2d, tries)
            mask = torch.nonzero(mask_2d.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        'Mask generator says: "Valid mask not found, decreasing acceptable-regions [%s]"',
                        tries,
                    )
        mask = mask.squeeze()
        mask_complement = 1 - mask_2d
        if log_detail:
            logger.info(
                "[blockwise] _build_blockwise_mask: num_patches=%s len(mask)=%s tries=%s",
                num_patches, len(mask), tries,
            )
        return mask, mask_complement

    def __call__(self, batch):
        B = len(batch)
        log_detail = self.debug_log and (getattr(self, "_debug_logged_batches", 0) == 0)
        if self.debug_log:
            logger.info("[blockwise] __call__: B=%s", B)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        num_pred_patches = self._sample_mask_budget(
            generator=g,
            scale=self.pred_mask_scale,
            label="pred",
            log_detail=log_detail,
        )
        num_enc_patches = self._sample_mask_budget(
            generator=g,
            scale=self.enc_mask_scale,
            label="enc",
            log_detail=log_detail,
        )
        if self.debug_log:
            logger.info(
                "[blockwise] __call__: num_pred_patches=%s num_enc_patches=%s",
                num_pred_patches, num_enc_patches,
            )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._build_blockwise_mask(
                    num_pred_patches,
                    aspect_ratio_range=self.aspect_ratio,
                    log_detail=log_detail,
                )
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning("Encountered exception in mask-generator %s", e)

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._build_blockwise_mask(
                    num_enc_patches,
                    aspect_ratio_range=(1.0, 1.0),
                    acceptable_regions=acceptable_regions,
                    log_detail=log_detail,
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        if self.debug_log:
            logger.info(
                "[blockwise] __call__: min_keep_pred=%s min_keep_enc=%s",
                min_keep_pred, min_keep_enc,
            )
        if log_detail:
            _k = 20
            logger.info(
                "[blockwise] collated_masks_pred length=%s shape[0]=%s sample [0][0,:%s]=%s",
                len(collated_masks_pred),
                tuple(collated_masks_pred[0].shape),
                _k,
                collated_masks_pred[0][0, :_k].tolist(),
            )
            logger.info(
                "[blockwise] collated_masks_enc length=%s shape[0]=%s sample [0][0,:%s]=%s",
                len(collated_masks_enc),
                tuple(collated_masks_enc[0].shape),
                _k,
                collated_masks_enc[0][0, :_k].tolist(),
            )
        if self.debug_log:
            self._debug_logged_batches += 1

        return collated_batch, collated_masks_enc, collated_masks_pred
