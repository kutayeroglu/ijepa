"""Grid masking collator for I-JEPA.

Keeps one patch out of every ``keep_stride`` x ``keep_stride`` cell as the
encoder context; every remaining patch forms the prediction target. With the
default ``keep_stride=2`` exactly one of every four patches is kept.

The mask is deterministic (no per-batch randomness), so every sample in a
batch receives the same regular grid of kept patches.
"""

from logging import getLogger
from multiprocessing import Value

import torch

logger = getLogger()


class MaskCollator(object):
    """Regular-grid mask: keep one patch per (stride x stride) cell."""

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        keep_stride=2,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.keep_stride = keep_stride
        self._itr_counter = Value('i', -1)  # collator is shared across workers

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def grid_masks(self):
        """Return ``(enc_keep_2d, pred_keep_2d)`` as ``[H, W]`` int tensors.

        ``enc_keep_2d`` marks the regularly spaced kept patches (context);
        ``pred_keep_2d`` is its complement (target).
        """
        s = self.keep_stride
        enc = torch.zeros(self.height, self.width, dtype=torch.int32)
        enc[::s, ::s] = 1
        pred = 1 - enc
        return enc, pred

    def __call__(self, batch):
        '''Collate imgs into a batch with encoder + predictor masks.'''
        collated_batch = torch.utils.data.default_collate(batch)

        enc_2d, pred_2d = self.grid_masks()
        enc_idx = torch.nonzero(enc_2d.flatten(), as_tuple=False).squeeze(1)
        pred_idx = torch.nonzero(pred_2d.flatten(), as_tuple=False).squeeze(1)

        B = len(batch)
        collated_masks_enc = torch.utils.data.default_collate(
            [[enc_idx] for _ in range(B)])
        collated_masks_pred = torch.utils.data.default_collate(
            [[pred_idx] for _ in range(B)])

        return collated_batch, collated_masks_enc, collated_masks_pred
