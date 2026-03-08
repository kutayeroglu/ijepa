#!/usr/bin/env python3
"""
Generates panel figures illustrating the mask selection process of
either MaskCollator variant overlaid on a real image:

  - **multinoise** (src/masks/multinoise.py) – spatially-structured
    dropout via colored-noise thresholding.
  - **multiblock** (src/masks/multiblock.py) – axis-aligned rectangular
    block masking (original I-JEPA strategy).
  - **compare** – both methods side by side on the same image(s).

Select the variant with ``--mask_type {multinoise,multiblock,compare}``.

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

Compare both methods on two images::

    python visualization/visualize_masks.py \
        --mask_type compare \
        --image_path photo1.jpg photo2.jpg

Custom multinoise scales::

    python visualization/visualize_masks.py \
        --image_path photo.jpg \
        --noise_path green_noise_data_3072.npz \
        --pred_mask_scale 0.10 0.25 \
        --enc_mask_scale  0.80 1.0  \
        --color_mask_ratio 0.4 \
        --seed 7

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
from datetime import datetime
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


def generate_masks(mask_type, *, input_size, patch_size, enc_mask_scale,
                   pred_mask_scale, aspect_ratio, npred, min_keep, seed,
                   noise_path=None, color_mask_ratio=0.15):
    """Sample context + target masks for a given mask type and seed.

    Returns ``(ctx_mask, target_masks)`` where *ctx_mask* is an ``[H, W]``
    binary tensor and *target_masks* is a list of ``[H, W]`` tensors.
    """
    H = W = input_size // patch_size

    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    if mask_type == 'multinoise':
        collator = MultinoiseCollator(
            input_size=(input_size, input_size),
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=1, npred=npred, min_keep=min_keep,
            allow_overlap=False,
            color_noise_path=noise_path,
            color_mask_ratio=color_mask_ratio,
        )
        p_size = collator._sample_block_size(
            generator=g, scale=collator.pred_mask_scale,
            aspect_ratio_scale=collator.aspect_ratio)
        e_size = collator._sample_block_size(
            generator=g, scale=collator.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0))

        rng_state = torch.get_rng_state()
        noise_grid = collator._extract_noise_windows(1)[0].cpu()
        torch.set_rng_state(rng_state)

        target_masks, complements = [], []
        for _ in range(npred):
            m, mc = collator._sample_noisy_block_mask(
                p_size, noise_grid=noise_grid)
            target_masks.append(_indices_to_2d(m, H, W))
            complements.append(mc)

        ctx_1d, _ = collator._sample_noisy_block_mask(
            e_size, noise_grid=noise_grid, acceptable_regions=complements)
        ctx_mask = _indices_to_2d(ctx_1d, H, W)

    else:  # multiblock
        collator = MultiblockCollator(
            input_size=(input_size, input_size),
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=1, npred=npred, min_keep=min_keep,
            allow_overlap=False,
        )
        p_size = collator._sample_block_size(
            generator=g, scale=collator.pred_mask_scale,
            aspect_ratio_scale=collator.aspect_ratio)
        e_size = collator._sample_block_size(
            generator=g, scale=collator.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0))

        target_masks, complements = [], []
        for _ in range(npred):
            m, mc = collator._sample_block_mask(p_size)
            target_masks.append(_indices_to_2d(m, H, W))
            complements.append(mc)

        ctx_1d, _ = collator._sample_block_mask(
            e_size, acceptable_regions=complements)
        ctx_mask = _indices_to_2d(ctx_1d, H, W)

    return ctx_mask, target_masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask_type', type=str, default='multinoise',
                        choices=['multinoise', 'multiblock', 'compare'],
                        help='Which mask collator to visualize '
                             '(compare = both methods on each image)')
    parser.add_argument('--image_path', type=str, nargs='+', required=True,
                        help='Path(s) to input image(s) (JPEG/PNG)')
    parser.add_argument('--noise_path', type=str,
                        default='green_noise_data_3072.npz',
                        help='Path to color-noise .npz file (multinoise only)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory; defaults to '
                             'visualization/<timestamp>_<mask_type>_seed<N>/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--pred_mask_scale', type=float, nargs=2,
                        default=[0.2, 0.3])
    parser.add_argument('--enc_mask_scale', type=float, nargs=2,
                        default=[0.85, 1.0])
    parser.add_argument('--aspect_ratio', type=float, nargs=2,
                        default=[0.75, 1.5])
    parser.add_argument('--color_mask_ratio', type=float, default=0.15,
                        help='Fraction of mask driven by color noise '
                             '(multinoise only)')
    parser.add_argument('--npred', type=int, default=4)
    parser.add_argument('--min_keep', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    # -- Output directory ---------------------------------------------------
    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = script_dir / f'{stamp}_{args.mask_type}_seed{args.seed}'
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = args.patch_size

    mask_kwargs = dict(
        input_size=args.input_size, patch_size=ps,
        enc_mask_scale=tuple(args.enc_mask_scale),
        pred_mask_scale=tuple(args.pred_mask_scale),
        aspect_ratio=tuple(args.aspect_ratio),
        npred=args.npred, min_keep=args.min_keep, seed=args.seed,
        noise_path=args.noise_path, color_mask_ratio=args.color_mask_ratio,
    )

    # -- Which methods to show ----------------------------------------------
    if args.mask_type == 'compare':
        mask_types = ['multiblock', 'multinoise']
    else:
        mask_types = [args.mask_type]

    # -- Load images --------------------------------------------------------
    image_entries = [(p, load_image(p, args.input_size))
                     for p in args.image_path]

    # -- Build rows: (row_label, panels) ------------------------------------
    rows = []
    for img_path, image in image_entries:
        for mt in mask_types:
            ctx_mask, target_masks = generate_masks(mt, **mask_kwargs)

            panels = [('Original', image),
                      ('Context', apply_patch_mask(image, ctx_mask, ps))]
            for t in range(args.npred):
                panels.append(
                    (f'Target {t + 1}',
                     apply_patch_mask(image, target_masks[t], ps)))

            rows.append((mt, panels))

    n_rows = len(rows)
    n_cols = len(rows[0][1])

    # -- Single-row: original behaviour + save individual panels ------------
    if n_rows == 1:
        _, panels = rows[0]

        for name, img_arr in panels:
            fname = name.lower().replace(' ', '')
            path = out_dir / f'{fname}.png'
            Image.fromarray(img_arr).save(str(path))
            print(f'Saved {path.resolve()}')

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        plt.subplots_adjust(wspace=0.06, left=0.005, right=0.995,
                            top=0.90, bottom=0.02)
        for ax, (name, img_arr) in zip(axes, panels):
            ax.imshow(img_arr)
            ax.set_title(name, fontsize=11, pad=6)
            ax.set_axis_off()

    # -- Multi-row comparison grid ------------------------------------------
    else:
        label_colors = {'multiblock': '#c0392b', 'multinoise': '#27ae60'}

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

        for row_idx, (mask_type_key, panels) in enumerate(rows):
            for col_idx, (name, img_arr) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                ax.imshow(img_arr)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(name, fontsize=14, pad=8)

            axes[row_idx, 0].text(
                -0.08, 0.5, mask_type_key.capitalize(),
                transform=axes[row_idx, 0].transAxes,
                va='center', ha='right',
                fontsize=15, fontweight='bold', rotation=90,
                color=label_colors.get(mask_type_key, 'black'))

        plt.subplots_adjust(wspace=0.06, hspace=0.10,
                            left=0.06, right=0.995,
                            top=0.94, bottom=0.02)

    # -- Save combined figure -----------------------------------------------
    for ext in ('png', 'pdf'):
        path = out_dir / f'combined.{ext}'
        fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
        print(f'Saved {path.resolve()}')

    plt.close(fig)


if __name__ == '__main__':
    main()
