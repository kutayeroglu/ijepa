"""Rasterized (quadrant) masking collator for I-JEPA.

Splits the patch grid into four quadrants. One quadrant (``context_quadrant``)
becomes the encoder context; the three remaining quadrants are the prediction
targets. Quadrants are indexed 0=top-left, 1=top-right, 2=bottom-left,
3=bottom-right.

The mask is deterministic (no per-batch randomness), so every sample in a
batch receives the same quadrant split.
"""

from logging import getLogger
from multiprocessing import Value

import torch

logger = getLogger()


class MaskCollator(object):
    """Quadrant mask: one quadrant is context, the other three are targets."""

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        context_quadrant=0,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.context_quadrant = context_quadrant % 4
        self._itr_counter = Value('i', -1)  # collator is shared across workers

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _quadrant_bounds(self):
        """Return ``(r0, r1, c0, c1)`` for quadrants 0=TL, 1=TR, 2=BL, 3=BR."""
        mh, mw = self.height // 2, self.width // 2
        return [
            (0, mh, 0, mw),
            (0, mh, mw, self.width),
            (mh, self.height, 0, mw),
            (mh, self.height, mw, self.width),
        ]

    def quadrant_masks(self):
        """Return ``(enc_keep_2d, pred_keep_2d_list)`` as ``[H, W]`` int tensors.

        ``enc_keep_2d`` is the context quadrant; ``pred_keep_2d_list`` holds the
        three remaining quadrants as separate targets.
        """
        masks = []
        for (r0, r1, c0, c1) in self._quadrant_bounds():
            m = torch.zeros(self.height, self.width, dtype=torch.int32)
            m[r0:r1, c0:c1] = 1
            masks.append(m)

        ctx_q = self.context_quadrant
        enc = masks[ctx_q]
        preds = [masks[i] for i in range(4) if i != ctx_q]
        return enc, preds

    def __call__(self, batch):
        '''Collate imgs into a batch with encoder + predictor masks.'''
        collated_batch = torch.utils.data.default_collate(batch)

        enc_2d, preds_2d = self.quadrant_masks()
        enc_idx = torch.nonzero(enc_2d.flatten(), as_tuple=False).squeeze(1)
        pred_idx = [torch.nonzero(p.flatten(), as_tuple=False).squeeze(1)
                    for p in preds_2d]

        B = len(batch)
        collated_masks_enc = torch.utils.data.default_collate(
            [[enc_idx] for _ in range(B)])
        collated_masks_pred = torch.utils.data.default_collate(
            [list(pred_idx) for _ in range(B)])

        return collated_batch, collated_masks_enc, collated_masks_pred
