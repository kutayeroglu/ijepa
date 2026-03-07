#!/usr/bin/env python3
"""
Generates a 6-panel figure illustrating the mask selection process of
either MaskCollator variant overlaid on a real image:

  - **multinoise** (src/masks/multinoise.py) – spatially-structured
    dropout via colored-noise thresholding.
  - **multiblock** (src/masks/multiblock.py) – axis-aligned rectangular
    block masking (original I-JEPA strategy).

Select the variant with ``--mask_type {multinoise,multiblock}``.

Panels (left to right):
  1. Original       – full input image
  2. Context        – only context-encoder patches visible
  3-6. Target 1--4  – only that target's patches visible

Non-selected patches are replaced with a neutral gray.

Usage
-----
Run from the project root. If ``src`` is not installed as a package,
set PYTHONPATH so that the imports resolve::

    export PYTHONPATH=/path/to/ijepa

Multinoise (default)::

    python visualization/visualize_masks.py --image_path photo.jpg

Multiblock::

    python visualization/visualize_masks.py \
        --mask_type multiblock \
        --image_path photo.jpg

Custom multinoise scales::

    python visualization/visualize_masks.py \
        --image_path photo.jpg \
        --pred_mask_scale 0.10 0.25 \
        --enc_mask_scale  0.80 1.0  \
        --color_mask_ratio 0.4 \
        --seed 7 \
        --output visualization/masks.png

Custom multiblock scales::

    python visualization/visualize_masks.py \
        --mask_type multiblock \
        --image_path photo.jpg \
        --pred_mask_scale 0.15 0.28 \
        --enc_mask_scale  0.85 1.0  \
        --aspect_ratio 0.75 1.5 \
        --seed 7
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.masks.multiblock import MaskCollator as MultiblockCollator
from src.masks.multinoise import MaskCollator as MultinoiseCollator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM_ALPHA = 0.25


def _indices_to_2d(mask_1d: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert 1D flattened indices back into an [H, W] binary mask."""
    flat = torch.zeros(H * W, dtype=torch.int32)
    flat[mask_1d] = 1
    return flat.reshape(H, W)


def apply_patch_mask(image: np.ndarray, mask_2d: torch.Tensor,
                     patch_size: int) -> np.ndarray:
    """Return a copy of *image* with non-selected patches dimmed."""
    out = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    H_p, W_p = mask_2d.shape
    for r in range(H_p):
        for c in range(W_p):
            if mask_2d[r, c]:
                r0, c0 = r * patch_size, c * patch_size
                out[r0:r0 + patch_size, c0:c0 + patch_size] = \
                    image[r0:r0 + patch_size, c0:c0 + patch_size]
    return out


def load_image(path: str, size: int) -> np.ndarray:
    """Load an image, resize/center-crop to *size* x *size*, return uint8 array."""
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
    ])
    img = transform(img)
    return np.array(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask_type', type=str, default='multinoise',
                        choices=['multinoise', 'multiblock'],
                        help='Which mask collator to visualize')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image (JPEG/PNG)')
    parser.add_argument('--noise_path', type=str,
                        default='green_noise_data_3072.npz',
                        help='Path to color-noise .npz file (multinoise only)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (pdf/png/svg); defaults to '
                             'visualization/mask_visualization_seed<N>.pdf')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--pred_mask_scale', type=float, nargs=2,
                        default=[0.15, 0.2])
    parser.add_argument('--enc_mask_scale', type=float, nargs=2,
                        default=[0.85, 1.0])
    parser.add_argument('--aspect_ratio', type=float, nargs=2,
                        default=[0.75, 1.5])
    parser.add_argument('--color_mask_ratio', type=float, default=0.3,
                        help='Fraction of mask driven by color noise (multinoise only)')
    parser.add_argument('--npred', type=int, default=4)
    parser.add_argument('--min_keep', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    if args.output is None:
        script_dir = Path(__file__).resolve().parent
        args.output = str(script_dir / f'mask_visualization_{args.mask_type}_seed{args.seed}.pdf')

    H = args.input_size // args.patch_size
    W = H
    ps = args.patch_size

    # -- Load image --------------------------------------------------------
    image = load_image(args.image_path, args.input_size)

    # -- Seed ---------------------------------------------------------------
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # -- Instantiate collator & sample masks --------------------------------
    if args.mask_type == 'multinoise':
        collator = MultinoiseCollator(
            input_size=(args.input_size, args.input_size),
            patch_size=ps,
            enc_mask_scale=tuple(args.enc_mask_scale),
            pred_mask_scale=tuple(args.pred_mask_scale),
            aspect_ratio=tuple(args.aspect_ratio),
            nenc=1,
            npred=args.npred,
            min_keep=args.min_keep,
            allow_overlap=False,
            color_noise_path=args.noise_path,
            color_mask_ratio=args.color_mask_ratio,
        )

        p_size = collator._sample_block_size(
            generator=g,
            scale=collator.pred_mask_scale,
            aspect_ratio_scale=collator.aspect_ratio,
        )
        e_size = collator._sample_block_size(
            generator=g,
            scale=collator.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )

        noise_grid = collator._extract_noise_windows(1)[0].cpu()

        target_masks = []
        complements = []
        for _ in range(args.npred):
            mask_1d, mask_complement = collator._sample_noisy_block_mask(
                p_size, noise_grid=noise_grid,
            )
            target_masks.append(_indices_to_2d(mask_1d, H, W))
            complements.append(mask_complement)

        ctx_1d, _ = collator._sample_noisy_block_mask(
            e_size, noise_grid=noise_grid, acceptable_regions=complements,
        )
        ctx_mask = _indices_to_2d(ctx_1d, H, W)

    else:  # multiblock
        collator = MultiblockCollator(
            input_size=(args.input_size, args.input_size),
            patch_size=ps,
            enc_mask_scale=tuple(args.enc_mask_scale),
            pred_mask_scale=tuple(args.pred_mask_scale),
            aspect_ratio=tuple(args.aspect_ratio),
            nenc=1,
            npred=args.npred,
            min_keep=args.min_keep,
            allow_overlap=False,
        )

        p_size = collator._sample_block_size(
            generator=g,
            scale=collator.pred_mask_scale,
            aspect_ratio_scale=collator.aspect_ratio,
        )
        e_size = collator._sample_block_size(
            generator=g,
            scale=collator.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )

        target_masks = []
        complements = []
        for _ in range(args.npred):
            mask_1d, mask_complement = collator._sample_block_mask(p_size)
            target_masks.append(_indices_to_2d(mask_1d, H, W))
            complements.append(mask_complement)

        ctx_1d, _ = collator._sample_block_mask(
            e_size, acceptable_regions=complements,
        )
        ctx_mask = _indices_to_2d(ctx_1d, H, W)

    # =====================================================================
    # Draw figure: 6 panels
    # =====================================================================
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    plt.subplots_adjust(wspace=0.06, left=0.005, right=0.995,
                        top=0.90, bottom=0.02)

    titles = ['Original', 'Context',
              'Target 1', 'Target 2', 'Target 3', 'Target 4']

    # Panel 1: original image
    axes[0].imshow(image)

    # Panel 2: context
    axes[1].imshow(apply_patch_mask(image, ctx_mask, ps))

    # Panels 3-6: targets
    for i in range(args.npred):
        axes[2 + i].imshow(apply_patch_mask(image, target_masks[i], ps))

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_axis_off()

    # ---- save ------------------------------------------------------------
    out = Path(args.output)
    fig.savefig(str(out), dpi=args.dpi, bbox_inches='tight')
    print(f'Saved {out.resolve()}')

    for ext in ('.pdf', '.png'):
        alt = out.with_suffix(ext)
        if alt != out:
            fig.savefig(str(alt), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {alt.resolve()}')

    plt.close(fig)


if __name__ == '__main__':
    main()
