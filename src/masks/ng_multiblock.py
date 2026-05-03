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


_VALID_BIAS = ("high", "low", "none")
_VALID_SCORE_MODE = ("boxsum", "corner")


class MaskCollator(object):
    """
    Noise-Guided MultiBlock collator.

    Identical to ``src.masks.multiblock.MaskCollator`` except that each
    block's top-left corner is drawn from a noise-weighted categorical
    distribution over all valid corners (Option A). The solid-rectangle
    structure of the masks is preserved, so the rectangle complement
    remains a clean constraint region for encoder/predictor separation.

    Behavioral guarantee: with ``enc_bias=pred_bias="none"`` the corner
    sampler falls back to the original ``torch.randint`` calls, making
    this collator bit-identical to vanilla multiblock for A/B tests.
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),     # total fraction of patches to mask
        pred_mask_scale=(0.2, 0.8),    # total fraction of patches to mask
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        debug_log=False,
        color_noise_path="noise_colors/green/green_noise_data_3072.npz",
        score_mode="boxsum",           # "boxsum" | "corner"
        enc_bias="high",               # "high" | "low" | "none"
        pred_bias="high",              # "high" | "low" | "none"
        noise_temperature=0.5,
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

        # -- Validate noise-bias hyperparameters
        if score_mode not in _VALID_SCORE_MODE:
            raise ValueError(
                f"score_mode must be one of {_VALID_SCORE_MODE}, got {score_mode!r}"
            )
        for name, val in [("enc_bias", enc_bias), ("pred_bias", pred_bias)]:
            if val not in _VALID_BIAS:
                raise ValueError(
                    f"{name} must be one of {_VALID_BIAS}, got {val!r}"
                )
        if float(noise_temperature) <= 0:
            raise ValueError(
                f"noise_temperature must be > 0, got {noise_temperature}"
            )
        self.score_mode = score_mode
        self.enc_bias = enc_bias
        self.pred_bias = pred_bias
        self.noise_temperature = float(noise_temperature)

        # -- Color Noise Initialization (mirrors src.masks.multinoise)
        self.trans_sequence = transforms.Compose([
            transforms.RandomCrop(self.height),  # Crop to [self.height, self.width]
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            NormalizeBySliceMax(),
        ])
        self._load_color_pattern(color_noise_path)

        if self.debug_log:
            logger.info(
                "[ng_multiblock] __init__: input_size=%s patch_size=%s height=%s width=%s "
                "enc_mask_scale=%s pred_mask_scale=%s aspect_ratio=%s nenc=%s npred=%s "
                "min_keep=%s allow_overlap=%s score_mode=%s enc_bias=%s pred_bias=%s "
                "noise_temperature=%s",
                input_size, patch_size, self.height, self.width,
                enc_mask_scale, pred_mask_scale, aspect_ratio, nenc, npred,
                min_keep, allow_overlap, score_mode, enc_bias, pred_bias,
                noise_temperature,
            )

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
                "[ng_multiblock] _sample_block_size(%s): scale=%s mask_scale=%s max_keep=%s "
                "aspect_ratio_scale=%s aspect_ratio=%s (h,w)=(%s,%s)",
                label, scale, mask_scale, max_keep, aspect_ratio_scale, aspect_ratio, h, w,
            )
        # e.g., if p_size = (8, 12), the block is 8 patches tall and 12 patches wide
        return (h, w)

    def _score_grid(
        self,
        noise_grid: torch.Tensor,
        h: int,
        w: int,
        mode: str,
    ) -> torch.Tensor:
        """
        Compute a score for every valid (top, left) corner of an h x w block.

        Returns a 2D tensor of shape [H - h, W - w], aligned with the same
        index range that vanilla multiblock samples uniformly via
        ``torch.randint(0, H - h)`` and ``torch.randint(0, W - w)``.

        - ``mode == "corner"``: score(t, l) = noise_grid[t, l]
        - ``mode == "boxsum"``: score(t, l) = sum of noise_grid over
          [t, t+h) x [l, l+w), computed in O(H*W) via an integral image
          (zero-padded on the top and left for clean indexing).
        """
        H, W = self.height, self.width
        if mode == "corner":
            return noise_grid[: H - h, : W - w]
        # -- boxsum via integral image
        ii = torch.zeros(H + 1, W + 1, dtype=noise_grid.dtype, device=noise_grid.device)
        ii[1:, 1:] = noise_grid.cumsum(0).cumsum(1)
        # box sum at (t, l) = ii[t+h, l+w] - ii[t, l+w] - ii[t+h, l] + ii[t, l]
        # for t in [0, H-h) and l in [0, W-w) (matches torch.randint(0, H-h) range)
        return (
            ii[h : H, w : W]
            - ii[: H - h, w : W]
            - ii[h : H, : W - w]
            + ii[: H - h, : W - w]
        )

    def _sample_corner(
        self,
        noise_grid: torch.Tensor | None,
        h: int,
        w: int,
        bias: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a single (top, left) corner.

        - ``bias == "none"`` (or noise_grid is None): uniform draw via
          ``torch.randint``, byte-for-byte the original multiblock path.
        - ``bias == "high"``: corners over noise-rich regions are favored.
        - ``bias == "low"``:  corners over noise-poor regions are favored.

        The score is min-max normalized to [0, 1] before applying
        softmax(score / T) so the temperature has the same operational
        meaning regardless of the noise grid's absolute scale or block size.
        """
        if bias == "none" or noise_grid is None:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            return top, left

        score = self._score_grid(noise_grid, h, w, self.score_mode)
        smin = score.min()
        smax = score.max()
        score_norm = (score - smin) / (smax - smin + 1e-8)
        if bias == "low":
            score_norm = 1.0 - score_norm
        logits = score_norm.flatten() / self.noise_temperature
        probs = torch.softmax(logits, dim=0)
        flat_idx = int(torch.multinomial(probs, 1).item())
        valid_W = self.width - w
        top = torch.tensor([flat_idx // valid_W])
        left = torch.tensor([flat_idx % valid_W])
        return top, left

    def _sample_block_mask(
        self,
        b_size: tuple[int, int],
        noise_grid: torch.Tensor | None = None,
        bias: str = "none",
        acceptable_regions: list[torch.Tensor] | None = None,
        log_detail: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = b_size  # block height and width (expressed in number of patches)

        def constrain_mask(mask, tries=0):
            """Restrict mask to the intersection of the first len-tries acceptable regions."""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]  # Element-wise: 1*1=1 (keep), 1*0=0 (remove)
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner (uniform if bias=="none", else noise-weighted)
            top, left = self._sample_corner(noise_grid, h, w, bias)
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1  # mark masked region as 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())  # Converts 2D binary mask to indices
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
        # Rectangle complement (identical to multiblock): the prediction
        # target is exactly the filled rectangle, so its complement is
        # safe to use as an acceptable_regions entry for encoder sampling.
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        if log_detail:
            top_val = top.item() if top.dim() > 0 else int(top)
            left_val = left.item() if left.dim() > 0 else int(left)
            logger.info(
                "[ng_multiblock] _sample_block_mask: b_size=%s top=%s left=%s len(mask)=%s tries=%s bias=%s",
                b_size, top_val, left_val, len(mask), tries, bias,
            )
        return mask, mask_complement

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch.
        # 1. sample enc/pred block sizes using a shared seed
        # 2. derive a per-image noise grid from the loaded NPZ pattern library
        # 3. sample several enc/pred block locations per image, with the
        #    top-left corner biased by the per-image noise grid (Option A)
        # 4. truncate to common length and collate
        """
        B = len(batch)
        log_detail = self.debug_log and (getattr(self, "_debug_logged_batches", 0) == 0)
        if self.debug_log:
            logger.info("[ng_multiblock] __call__: B=%s", B)

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
        if self.debug_log:
            logger.info("[ng_multiblock] __call__: p_size=%s", p_size)
        e_size = self._sample_block_size(  # block size for context block [h,w (in number of patches)]
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
            label="enc",
            log_detail=log_detail,
        )
        if self.debug_log:
            logger.info("[ng_multiblock] __call__: e_size=%s", e_size)

        # Per-image augmented noise grids, only consulted when at least one
        # role is biased. Avoids paying the augmentation cost in the
        # bit-identical fallback case (enc_bias == pred_bias == "none").
        need_noise = (self.enc_bias != "none") or (self.pred_bias != "none")
        batch_noise_grids = self._extract_noise_windows(B) if need_noise else None

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width  # safe upper bound
        min_keep_enc = self.height * self.width  # safe upper bound
        for i in range(B):
            noise_grid = None if batch_noise_grids is None else batch_noise_grids[i]

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(
                    p_size,
                    noise_grid=noise_grid,
                    bias=self.pred_bias,
                    log_detail=log_detail,
                )
                masks_p.append(mask)  # target block mask
                masks_C.append(mask_C)  # target block complement mask
                min_keep_pred = min(min_keep_pred, len(mask))
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
                    e_size,
                    noise_grid=noise_grid,
                    bias=self.enc_bias,
                    acceptable_regions=acceptable_regions,
                    log_detail=log_detail,
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # ensure all masks are the same size (truncate to minimum)
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
                "[ng_multiblock] __call__: min_keep_pred=%s min_keep_enc=%s",
                min_keep_pred, min_keep_enc,
            )
        if log_detail:
            _k = 20
            logger.info(
                "[ng_multiblock] collated_masks_pred length=%s shape[0]=%s sample [0][0,:%s]=%s",
                len(collated_masks_pred),
                tuple(collated_masks_pred[0].shape),
                _k,
                collated_masks_pred[0][0, :_k].tolist(),
            )
            logger.info(
                "[ng_multiblock] collated_masks_enc length=%s shape[0]=%s sample [0][0,:%s]=%s",
                len(collated_masks_enc),
                tuple(collated_masks_enc[0].shape),
                _k,
                collated_masks_enc[0][0, :_k].tolist(),
            )
        if self.debug_log:
            self._debug_logged_batches += 1

        return collated_batch, collated_masks_enc, collated_masks_pred
