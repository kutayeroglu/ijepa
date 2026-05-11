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

Use Turkish labels in the output figure::

    python visualization/visualize_masks.py \
        --image_path photo.jpg \
        --turkish

Mechanic 1 (patch-grid overlay only, no masks)::

    python visualization/visualize_masks.py \
        --figure patch_grid \
        --image_path photo.jpg

Mechanic 2 (scale + aspect-ratio sweep)::

    python visualization/visualize_masks.py \
        --figure block_size \
        --image_path photo.jpg \
        --scale_sweep 0.15 0.4 0.8 \
        --ar_sweep 0.5 1.0 2.0 \
        --fixed_scale_for_ar 0.4
"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches as mpatches
from PIL import Image
from torchvision import transforms as T

from src.masks.multiblock import MaskCollator as MultiblockCollator
from src.masks.multinoise import MaskCollator as MultinoiseCollator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM_ALPHA = 0.25
TR_MASK_TYPE_LABELS = {
    'multinoise': 'Çoklu Gürültü',
    'multiblock': 'Çoklu Blok',
}


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


def localized_labels(turkish: bool):
    """Return panel labels for the selected output language."""
    if turkish:
        return {
            'original': 'Girdi',
            'context': 'Bağlam',
            'target_prefix': 'Hedef',
        }
    return {
        'original': 'Original',
        'context': 'Context',
        'target_prefix': 'Target',
    }


def localized_grid_labels(turkish: bool):
    """Return panel labels for the patch-grid figure (Mechanic 1)."""
    if turkish:
        return {
            'input': 'Girdi görüntüsü',
            'grid_title_fmt': '{n}×{n} yama ızgarası',
        }
    return {
        'input': 'Input image',
        'grid_title_fmt': '{n}×{n} patch grid',
    }


def localized_block_size_labels(turkish: bool):
    """Return panel labels for the block-size figure (Mechanic 2)."""
    if turkish:
        return {
            'title_fmt': ('ölçek = {s:.2f}, en/boy = {ar:.2f}\n'
                          '(y, g) = ({h}, {w}), {n} yama'),
            'row_scale': 'Ölçek alanı belirler',
            'row_ar': 'En/boy oranı şekli belirler',
            'caption': ('Hedef bloğu: aspect_ratio ∈ (0.75, 1.5);  '
                        'bağlam bloğu: aspect_ratio = 1.0 (kare).'),
        }
    return {
        'title_fmt': ('scale = {s:.2f}, ar = {ar:.2f}\n'
                      '(h, w) = ({h}, {w}), {n} patches'),
        'row_scale': 'Scale controls area',
        'row_ar': 'Aspect ratio controls shape',
        'caption': ('Target block: aspect_ratio ∈ (0.75, 1.5);  '
                    'context block: aspect_ratio = 1.0 (square).'),
    }


def _draw_grid_lines(ax, image: np.ndarray, patch_size: int,
                     color: str = 'white', lw: float = 0.6,
                     alpha: float = 0.85):
    """Show *image* on *ax* and draw the patch-boundary grid on top.

    Grid lines are offset by 0.5 px to align with ``imshow``'s pixel
    boundaries (matplotlib places pixel ``n`` centered at ``n``).
    """
    H, W = image.shape[:2]
    n_h, n_w = H // patch_size, W // patch_size
    ax.imshow(image)
    for k in range(n_h + 1):
        ax.axhline(k * patch_size - 0.5, color=color, lw=lw, alpha=alpha)
    for k in range(n_w + 1):
        ax.axvline(k * patch_size - 0.5, color=color, lw=lw, alpha=alpha)


def draw_patch_grid_panel(ax, image: np.ndarray, patch_size: int,
                          highlight_patch=(0, 0), turkish: bool = False):
    """Plot *image* on *ax* with a patch-grid overlay and one highlighted cell."""
    _draw_grid_lines(ax, image, patch_size)

    r, c = highlight_patch
    ax.add_patch(mpatches.Rectangle(
        (c * patch_size - 0.5, r * patch_size - 0.5),
        patch_size, patch_size,
        fill=False, edgecolor='red', linewidth=2.0))

    text = (f'1 yama\n= {patch_size}×{patch_size} piksel'
            if turkish else f'1 patch\n= {patch_size}×{patch_size} px')
    ax.annotate(
        text,
        xy=((c + 1) * patch_size, (r + 1) * patch_size),
        xytext=(c * patch_size + 55, r * patch_size + 55),
        color='red', fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    ax.set_axis_off()


def compute_block_hw(grid_h: int, grid_w: int,
                     scale: float, aspect_ratio: float) -> tuple[int, int]:
    """Deterministic mirror of ``MaskCollator._sample_block_size``.

    Reproduces the (h, w) formula used by the multiblock collator
    (``src/masks/multiblock.py``, lines ~75–86) without random sampling.
    """
    max_keep = int(grid_h * grid_w * scale)
    h = int(round(math.sqrt(max_keep * aspect_ratio)))
    w = int(round(math.sqrt(max_keep / aspect_ratio)))
    while h >= grid_h:
        h -= 1
    while w >= grid_w:
        w -= 1
    return h, w


def draw_block_panel(ax, image: np.ndarray, patch_size: int,
                     block_h: int, block_w: int,
                     anchor: tuple[int, int] | None = None,
                     color: str = 'red',
                     fill_alpha: float = 0.4):
    """Dim *image*, draw patch grid, overlay a block of (block_h, block_w) patches.

    ``anchor`` is either ``None`` (center the block in the grid) or an
    explicit ``(top_row, left_col)`` tuple in patch coordinates.
    """
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    grid_h = image.shape[0] // patch_size
    grid_w = image.shape[1] // patch_size

    if anchor is None:
        top = max((grid_h - block_h) // 2, 0)
        left = max((grid_w - block_w) // 2, 0)
    else:
        top, left = anchor

    ax.add_patch(mpatches.Rectangle(
        (left * patch_size - 0.5, top * patch_size - 0.5),
        block_w * patch_size, block_h * patch_size,
        facecolor=color, alpha=fill_alpha,
        edgecolor=color, linewidth=2.0))
    ax.set_axis_off()


def localized_mask_type_label(mask_type: str, turkish: bool) -> str:
    """Return display label for the selected mask type."""
    if turkish:
        return TR_MASK_TYPE_LABELS.get(mask_type, mask_type)
    return mask_type.capitalize()


def generate_masks(mask_type, *, input_size, patch_size, enc_mask_scale,
                   pred_mask_scale, aspect_ratio, npred, min_keep, seed,
                   noise_path='green_noise_data_3072.npz',
                   color_mask_ratio=0.15,
                   enc_drop_order="lowest", pred_drop_order="lowest"):
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
            enc_drop_order=enc_drop_order,
            pred_drop_order=pred_drop_order,
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
                p_size, noise_grid=noise_grid, drop_order=pred_drop_order)
            target_masks.append(_indices_to_2d(m, H, W))
            complements.append(mc)

        ctx_1d, _ = collator._sample_noisy_block_mask(
            e_size, noise_grid=noise_grid, acceptable_regions=complements,
            drop_order=enc_drop_order)
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
    parser.add_argument('--figure', type=str, default='masks',
                        choices=['masks', 'patch_grid', 'block_size'],
                        help='Which figure to generate: '
                             '"masks" (current behavior, mask panels), '
                             '"patch_grid" (Mechanic 1: image + grid overlay), '
                             'or "block_size" (Mechanic 2: scale + aspect-ratio '
                             'sweep)')
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
    parser.add_argument('--enc_drop_order', type=str, default='lowest',
                        choices=['lowest', 'highest'],
                        help='Which noise-value patches to drop for context '
                             '(multinoise only): lowest or highest')
    parser.add_argument('--pred_drop_order', type=str, default='lowest',
                        choices=['lowest', 'highest'],
                        help='Which noise-value patches to drop for target '
                             '(multinoise only): lowest or highest')
    parser.add_argument('--npred', type=int, default=4)
    parser.add_argument('--min_keep', type=int, default=10)
    parser.add_argument('--scale_sweep', type=float, nargs=3,
                        default=[0.15, 0.4, 0.8],
                        help='Three scale values for the Mechanic 2 scale-sweep '
                             'row (figure=block_size only)')
    parser.add_argument('--ar_sweep', type=float, nargs=3,
                        default=[0.5, 1.0, 2.0],
                        help='Three aspect-ratio values for the Mechanic 2 '
                             'ar-sweep row (figure=block_size only)')
    parser.add_argument('--fixed_scale_for_ar', type=float, default=0.4,
                        help='Scale held fixed in the aspect-ratio sweep row '
                             '(figure=block_size only)')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--turkish', action='store_true',
                        help='Use Turkish labels in panel titles '
                             'and method names')
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

    # -- Mechanic 1: image + patch-grid overlay -----------------------------
    if args.figure == 'patch_grid':
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        n_grid = args.input_size // ps
        g_labels = localized_grid_labels(args.turkish)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image)
        axes[0].set_title(g_labels['input'], fontsize=12, pad=6)
        axes[0].set_axis_off()

        draw_patch_grid_panel(axes[1], image, ps, turkish=args.turkish)
        axes[1].set_title(g_labels['grid_title_fmt'].format(n=n_grid),
                          fontsize=12, pad=6)

        plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99,
                            top=0.92, bottom=0.02)

        for ext in ('png', 'pdf'):
            path = out_dir / f'mechanic1_patch_grid.{ext}'
            fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {path.resolve()}')
        plt.close(fig)
        return

    # -- Mechanic 2: scale & aspect-ratio sweep -----------------------------
    if args.figure == 'block_size':
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        grid_h = grid_w = args.input_size // ps
        bs_labels = localized_block_size_labels(args.turkish)

        fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))

        for col, s in enumerate(args.scale_sweep):
            h, w = compute_block_hw(grid_h, grid_w, s, 1.0)
            draw_block_panel(axes[0, col], image, ps, h, w)
            axes[0, col].set_title(
                bs_labels['title_fmt'].format(
                    s=s, ar=1.0, h=h, w=w, n=h * w),
                fontsize=10, pad=4)

        s_fixed = args.fixed_scale_for_ar
        for col, ar in enumerate(args.ar_sweep):
            h, w = compute_block_hw(grid_h, grid_w, s_fixed, ar)
            draw_block_panel(axes[1, col], image, ps, h, w)
            axes[1, col].set_title(
                bs_labels['title_fmt'].format(
                    s=s_fixed, ar=ar, h=h, w=w, n=h * w),
                fontsize=10, pad=4)

        axes[0, 0].text(
            -0.06, 0.5, bs_labels['row_scale'],
            transform=axes[0, 0].transAxes,
            va='center', ha='right',
            fontsize=12, fontweight='bold', rotation=90, color='#2c3e50')
        axes[1, 0].text(
            -0.06, 0.5, bs_labels['row_ar'],
            transform=axes[1, 0].transAxes,
            va='center', ha='right',
            fontsize=12, fontweight='bold', rotation=90, color='#2c3e50')

        fig.text(0.5, 0.02, bs_labels['caption'],
                 ha='center', fontsize=10, style='italic', color='#34495e')

        plt.subplots_adjust(wspace=0.06, hspace=0.18,
                            left=0.06, right=0.99,
                            top=0.94, bottom=0.08)

        for ext in ('png', 'pdf'):
            path = out_dir / f'mechanic2_block_size.{ext}'
            fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {path.resolve()}')
        plt.close(fig)
        return

    mask_kwargs = dict(
        input_size=args.input_size, patch_size=ps,
        enc_mask_scale=tuple(args.enc_mask_scale),
        pred_mask_scale=tuple(args.pred_mask_scale),
        aspect_ratio=tuple(args.aspect_ratio),
        npred=args.npred, min_keep=args.min_keep, seed=args.seed,
        noise_path=args.noise_path, color_mask_ratio=args.color_mask_ratio,
        enc_drop_order=args.enc_drop_order, pred_drop_order=args.pred_drop_order,
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
    labels = localized_labels(args.turkish)
    rows = []
    for img_path, image in image_entries:
        for mt in mask_types:
            ctx_mask, target_masks = generate_masks(mt, **mask_kwargs)

            panels = [(labels['original'], image),
                      (labels['context'], apply_patch_mask(image, ctx_mask, ps))]
            for t in range(args.npred):
                panels.append(
                    (f"{labels['target_prefix']} {t + 1}",
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
                -0.08, 0.5, localized_mask_type_label(mask_type_key, args.turkish),
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
