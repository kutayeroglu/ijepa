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

Mechanic 3 (corner sampling: valid region + samples)::

    python visualization/visualize_masks.py \
        --figure placement \
        --image_path photo.jpg \
        --placement_block_scale 0.2 \
        --placement_block_ar 1.0 \
        --n_placements 4

Mechanic 3.5 (noise-guided patch removal — multinoise only)::

    python visualization/visualize_masks.py \
        --figure noise_dropout \
        --image_path photo.jpg \
        --noise_path green_noise_data_3072.npz \
        --noise_block_scale 0.30 \
        --color_mask_ratio 0.15

Noise-dropout ColormAE (full-grid: noise overlay then drop lowest-noise
patches; default 75% dropped, no rectangular block sampling)::

    python visualization/visualize_masks.py \
        --figure noise_dropout_colormae \
        --image_path photo.jpg \
        --noise_path green_noise_data_3072.npz \
        --colormae_drop_ratio 0.75

Mechanic 3.6 (noise map transformation pipeline)::

    python visualization/visualize_masks.py \
        --figure noise_transform \
        --image_path photo.jpg \
        --noise_path green_noise_data_3072.npz \
        --seed 42

Mechanic 4 (carving trick: target/context overlap removal)::

    python visualization/visualize_masks.py \
        --figure carving \
        --image_path photo.jpg \
        --npred 4 \
        --carving_pred_scale 0.10 \
        --carving_enc_scale 0.55

Mechanic 4b (two rows: multiblock rectangles vs multinoise with the same
sampled placements; left margin labels name each row)::

    python visualization/visualize_masks.py \
        --figure carving_extended \
        --image_path photo.jpg \
        --noise_path green_noise_data_3072.npz \
        --npred 4 \
        --carving_pred_scale 0.10 \
        --carving_enc_scale 0.55 \
        --color_mask_ratio 0.15
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
from src.masks.multinoise import MaskCollator as MultinoiseCollator, NormalizeBySliceMax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM_ALPHA = 0.40
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


def localized_placement_labels(turkish: bool):
    """Return panel labels for the placement figure (Mechanic 3)."""
    if turkish:
        return {
            'block_title_fmt': 'Blok şekli: (y, g) = ({h}, {w})',
            'region_title': 'Geçerli köşe bölgesi',
            'region_caption_fmt': ('Geçerli sol-üst köşeler:\n'
                                   '(Y−y)×(G−g) = {a}×{b} = {n} konum'),
            'single_title': 'Tek örnekleme',
            'single_caption_fmt': 'köşe = ({r}, {c})',
            'multi_title_fmt': '{n} adet düzgün örnek',
        }
    return {
        'block_title_fmt': 'Block shape: (h, w) = ({h}, {w})',
        'region_title': 'Valid corner region',
        'region_caption_fmt': ('Valid top-left corners:\n'
                               '(H−h)×(W−w) = {a}×{b} = {n} positions'),
        'single_title': 'One sample',
        'single_caption_fmt': 'corner = ({r}, {c})',
        'multi_title_fmt': '{n} uniform samples',
    }


def _add_valid_region_patch(ax, patch_size: int,
                            valid_h: int, valid_w: int,
                            region_color: str = '#27ae60',
                            fill_alpha: float = 0.35):
    """Add the (H-h)×(W-w) valid-corner rectangle to an existing axes."""
    ax.add_patch(mpatches.Rectangle(
        (-0.5, -0.5),
        valid_w * patch_size, valid_h * patch_size,
        facecolor=region_color, alpha=fill_alpha,
        edgecolor=region_color, linewidth=2.0))


def draw_allowed_region_panel(ax, image: np.ndarray, patch_size: int,
                              block_h: int, block_w: int,
                              region_color: str = '#27ae60',
                              fill_alpha: float = 0.35):
    """Show the rectangle of valid top-left corners (size (H-h)×(W-w))."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    grid_h = image.shape[0] // patch_size
    grid_w = image.shape[1] // patch_size
    valid_h = max(grid_h - block_h, 0)
    valid_w = max(grid_w - block_w, 0)

    _add_valid_region_patch(ax, patch_size, valid_h, valid_w,
                            region_color=region_color,
                            fill_alpha=fill_alpha)
    ax.set_axis_off()
    return valid_h, valid_w


def draw_single_placement_panel(ax, image: np.ndarray, patch_size: int,
                                block_h: int, block_w: int,
                                top: int, left: int,
                                color: str = 'red',
                                show_valid_region: bool = True):
    """Show one sampled corner (dot) and the resulting block outline.

    When ``show_valid_region`` is True, the same green ``(H-h)×(W-w)``
    rectangle from the allowed-region panel is overlaid so the viewer
    can see which corner-region position was picked. The block itself
    is drawn outline-only (no fill) to keep the image visible underneath.
    """
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    if show_valid_region:
        grid_h = image.shape[0] // patch_size
        grid_w = image.shape[1] // patch_size
        valid_h = max(grid_h - block_h, 0)
        valid_w = max(grid_w - block_w, 0)
        _add_valid_region_patch(ax, patch_size, valid_h, valid_w)

    ax.add_patch(mpatches.Rectangle(
        (left * patch_size - 0.5, top * patch_size - 0.5),
        block_w * patch_size, block_h * patch_size,
        fill=False, edgecolor=color, linewidth=2.2))

    ax.plot(left * patch_size - 0.5, top * patch_size - 0.5,
            'o', color=color, markersize=9,
            markeredgecolor='white', markeredgewidth=1.5,
            zorder=5)
    ax.set_axis_off()


def draw_multi_placement_panel(ax, image: np.ndarray, patch_size: int,
                               block_h: int, block_w: int,
                               placements: list[tuple[int, int]],
                               colors: list[str] | None = None,
                               linewidth: float = 2.0):
    """Show N sampled blocks overlaid in different colors (outlines only)."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    if colors is None:
        colors = ['#e74c3c', '#3498db', '#f1c40f', '#9b59b6',
                  '#1abc9c', '#e67e22']

    for i, (top, left) in enumerate(placements):
        c = colors[i % len(colors)]
        ax.add_patch(mpatches.Rectangle(
            (left * patch_size - 0.5, top * patch_size - 0.5),
            block_w * patch_size, block_h * patch_size,
            fill=False, edgecolor=c, linewidth=linewidth))
    ax.set_axis_off()


def sample_placements(grid_h: int, grid_w: int,
                      block_h: int, block_w: int,
                      n: int, generator: torch.Generator
                      ) -> list[tuple[int, int]]:
    """Sample N uniform (top, left) corners — mirrors multiblock's randint calls."""
    placements: list[tuple[int, int]] = []
    high_h = max(grid_h - block_h, 1)
    high_w = max(grid_w - block_w, 1)
    for _ in range(n):
        top = int(torch.randint(0, high_h, (1,), generator=generator).item())
        left = int(torch.randint(0, high_w, (1,), generator=generator).item())
        placements.append((top, left))
    return placements


def localized_carving_labels(turkish: bool):
    """Return panel labels for the carving figure (  4)."""
    if turkish:
        return {
            'targets': '(a) Hedef bloklar',
            'acceptable': '(b) Kabul edilebilir bölge',
            'candidate': '(c) Aday bağlam bloku',
            'final': '(d) Yontulmuş nihai bağlam',
            'caption': ('nihai = aday  ⊙  ⋂ᵢ tümleyen(Tᵢ)  '
                        '=  aday  ∩  kabul edilebilir bölge'),
        }
    return {
        'targets': '(a) Target blocks',
        'acceptable': '(b) Acceptable region (complement)',
        'candidate': '(c) Candidate context block',
        'final': '(d) Final carved context',
        'caption': ('final = candidate  ⊙  ⋂ᵢ complement(Tᵢ)  '
                    '=  candidate  ∩  acceptable region'),
    }


def localized_carving_extended_labels(turkish: bool):
    """Left-side row labels for the two-row carving_extended figure."""
    if turkish:
        return {
            'side_multiblock': 'Çoklu blok',
            'side_multinoise': 'Çoklu gürültü',
        }
    return {
        'side_multiblock': 'Multiblock',
        'side_multinoise': 'Multinoise',
    }


def compute_multinoise_carving_state(
        grid_h: int, grid_w: int,
        targets: list[tuple[int, int, int, int]],
        candidate: tuple[int, int, int, int],
        noise_grid: np.ndarray,
        color_mask_ratio: float,
        pred_drop_order: str,
        enc_drop_order: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Fixed placements: reproduce multinoise target complements + context.

    For each target rectangle, apply ``apply_noise_threshold`` (predictor
    drop order).  Complements follow the collator: ``1 - mask_2d`` on the
    **kept** mask.  Acceptable is the element-wise AND of complements.

    Context matches ``_sample_noisy_block_mask``: threshold noise on the
    full candidate rectangle first, then multiply by each complement.

    Returns
    -------
    target_kepts
        List of ``[H, W]`` bool — kept predictor patches per target.
    acceptable
        ``[H, W]`` bool — AND of complements (where context may overlap).
    cand_after_noise
        Kept patches after noise only (inside candidate rectangle).
    final_carved
        ``cand_after_noise & acceptable``.
    """
    target_kepts: list[np.ndarray] = []
    complements: list[np.ndarray] = []
    for (top, left, h, w) in targets:
        rect = np.zeros((grid_h, grid_w), dtype=bool)
        rect[top:top + h, left:left + w] = True
        kept, _ = apply_noise_threshold(
            rect, noise_grid, color_mask_ratio,
            drop_order=pred_drop_order)
        kept = kept.astype(bool)
        target_kepts.append(kept)
        complements.append(~kept)

    acceptable = np.ones((grid_h, grid_w), dtype=bool)
    for comp in complements:
        acceptable &= comp.astype(bool)

    ct, cl, ch, cw = candidate
    cand_block = np.zeros((grid_h, grid_w), dtype=bool)
    cand_block[ct:ct + ch, cl:cl + cw] = True
    cand_after_noise, _ = apply_noise_threshold(
        cand_block, noise_grid, color_mask_ratio,
        drop_order=enc_drop_order)
    cand_after_noise = cand_after_noise.astype(bool)
    final_carved = cand_after_noise & acceptable
    return target_kepts, acceptable, cand_after_noise, final_carved


def _union_kept_masks(kept_list: list[np.ndarray]) -> np.ndarray:
    u = np.zeros_like(kept_list[0], dtype=bool)
    for m in kept_list:
        u |= m
    return u


def draw_multinoise_targets_kept_panel(
        ax, image: np.ndarray, patch_size: int,
        target_kepts: list[np.ndarray],
        targets: list[tuple[int, int, int, int]],
        fill_colors: list[tuple[float, float, float, float]] | None = None):
    """Panel (a′): dimmed grid + per-target kept patches + dashed bbox."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)
    if fill_colors is None:
        fill_colors = [
            (0.75, 0.22, 0.17, 0.52),
            (0.20, 0.45, 0.75, 0.48),
            (0.55, 0.35, 0.75, 0.45),
            (0.15, 0.65, 0.50, 0.45),
            (0.85, 0.55, 0.15, 0.45),
            (0.45, 0.20, 0.55, 0.45),
        ]
    for i, kept in enumerate(target_kepts):
        rgba = fill_colors[i % len(fill_colors)]
        _add_per_patch_overlay(ax, patch_size, kept, rgba)
    dash = (0, (3, 3))
    for i, (top, left, h, w) in enumerate(targets):
        ax.add_patch(mpatches.Rectangle(
            (left * patch_size - 0.5, top * patch_size - 0.5),
            w * patch_size, h * patch_size,
            fill=False, edgecolor='#7f8c8d', linewidth=1.2,
            linestyle=dash, zorder=2))
        ax.text(
            (left + w / 2) * patch_size - 0.5,
            (top + h / 2) * patch_size - 0.5,
            f'T{i + 1}', color='#2c3e50', fontsize=10, fontweight='bold',
            ha='center', va='center', zorder=4,
            bbox=dict(facecolor='white', alpha=0.82,
                      edgecolor='none', pad=1.5))
    ax.set_axis_off()


def draw_multinoise_acceptable_kept_panel(
        ax, image: np.ndarray, patch_size: int,
        target_kepts: list[np.ndarray],
        targets: list[tuple[int, int, int, int]],
        target_color: str = '#c0392b',
        accept_rgba: tuple[float, float, float, float] = (
            0.15, 0.68, 0.38, 0.40),
        block_rgba: tuple[float, float, float, float] = (
            0.75, 0.22, 0.17, 0.30)):
    """Panel (b′): green outside union of *kept* targets; union tinted red."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)
    union_k = _union_kept_masks(target_kepts)
    _add_per_patch_overlay(ax, patch_size, ~union_k, accept_rgba)
    _add_per_patch_overlay(ax, patch_size, union_k, block_rgba)
    labels = [f'T{i + 1}' for i in range(len(targets))]
    _add_block_outlines(ax, patch_size, targets,
                        color=target_color, linewidth=1.4,
                        labels=labels)
    ax.set_axis_off()


def draw_multinoise_candidate_kept_panel(
        ax, image: np.ndarray, patch_size: int,
        target_kepts: list[np.ndarray],
        targets: list[tuple[int, int, int, int]],
        candidate: tuple[int, int, int, int],
        target_color: str = '#c0392b',
        candidate_color: str = '#2980b9'):
    """Panel (c′): acceptable-from-kept + candidate rectangle outline."""
    draw_multinoise_acceptable_kept_panel(
        ax, image, patch_size, target_kepts, targets,
        target_color=target_color)
    _add_block_outlines(ax, patch_size, [candidate],
                        color=candidate_color, linewidth=2.5,
                        labels=['C'])
    ax.set_axis_off()


def draw_multinoise_final_kept_panel(
        ax, image: np.ndarray, patch_size: int,
        targets: list[tuple[int, int, int, int]],
        candidate: tuple[int, int, int, int],
        final_mask: np.ndarray,
        target_color: str = '#c0392b',
        candidate_color: str = '#2980b9',
        final_rgba: tuple[float, float, float, float] = (
            0.16, 0.50, 0.73, 0.55)):
    """Panel (d′): carved context after noise then complement multiply."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)
    _add_per_patch_overlay(ax, patch_size, final_mask, final_rgba)
    labels = [f'T{i + 1}' for i in range(len(targets))]
    _add_block_outlines(ax, patch_size, targets,
                        color=target_color, linewidth=1.4,
                        labels=labels)
    _add_block_outlines(ax, patch_size, [candidate],
                        color=candidate_color, linewidth=2.0,
                        labels=['C'])
    ax.set_axis_off()


def _compute_union_mask(grid_h: int, grid_w: int,
                        rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Return a (grid_h, grid_w) bool mask: True where any rect covers."""
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for (top, left, h, w) in rects:
        mask[top:top + h, left:left + w] = True
    return mask


def _add_per_patch_overlay(ax, patch_size: int,
                           mask_2d: np.ndarray,
                           rgba: tuple[float, float, float, float]):
    """Tint every True patch in *mask_2d* with the given RGBA colour."""
    mask_full = np.repeat(np.repeat(mask_2d, patch_size, axis=0),
                          patch_size, axis=1)
    overlay = np.zeros((*mask_full.shape, 4), dtype=np.float32)
    overlay[mask_full] = rgba
    ax.imshow(overlay)


def _add_block_outlines(ax, patch_size: int,
                        rects: list[tuple[int, int, int, int]],
                        color: str = '#c0392b',
                        linewidth: float = 2.0,
                        labels: list[str] | None = None):
    """Add outline rectangles for each (top, left, h, w) in *rects*."""
    for i, (top, left, h, w) in enumerate(rects):
        ax.add_patch(mpatches.Rectangle(
            (left * patch_size - 0.5, top * patch_size - 0.5),
            w * patch_size, h * patch_size,
            fill=False, edgecolor=color, linewidth=linewidth, zorder=3))
        if labels is not None and i < len(labels):
            ax.text(
                (left + w / 2) * patch_size - 0.5,
                (top + h / 2) * patch_size - 0.5,
                labels[i], color=color, fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=4,
                bbox=dict(facecolor='white', alpha=0.85,
                          edgecolor='none', pad=1.8))


def draw_targets_only_panel(ax, image: np.ndarray, patch_size: int,
                            targets: list[tuple[int, int, int, int]],
                            target_color: str = '#c0392b'):
    """Panel (a): dimmed image + N target outlines + Tᵢ labels."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)
    labels = [f'T{i + 1}' for i in range(len(targets))]
    _add_block_outlines(ax, patch_size, targets,
                        color=target_color, labels=labels)
    ax.set_axis_off()


def draw_acceptable_panel(ax, image: np.ndarray, patch_size: int,
                          targets: list[tuple[int, int, int, int]],
                          target_color: str = '#c0392b',
                          accept_rgba: tuple[float, float, float, float] = (
                              0.15, 0.68, 0.38, 0.40),
                          block_rgba: tuple[float, float, float, float] = (
                              0.75, 0.22, 0.17, 0.30),
                          with_target_outlines: bool = True):
    """Panel (b): green tint on acceptable patches, red tint on target patches."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    grid_h = image.shape[0] // patch_size
    grid_w = image.shape[1] // patch_size
    target_mask = _compute_union_mask(grid_h, grid_w, targets)

    _add_per_patch_overlay(ax, patch_size, ~target_mask, accept_rgba)
    _add_per_patch_overlay(ax, patch_size, target_mask, block_rgba)

    if with_target_outlines:
        labels = [f'T{i + 1}' for i in range(len(targets))]
        _add_block_outlines(ax, patch_size, targets,
                            color=target_color, labels=labels)
    ax.set_axis_off()


def draw_candidate_panel(ax, image: np.ndarray, patch_size: int,
                         targets: list[tuple[int, int, int, int]],
                         candidate: tuple[int, int, int, int],
                         target_color: str = '#c0392b',
                         candidate_color: str = '#2980b9'):
    """Panel (c): acceptable-region panel + candidate context block outline."""
    draw_acceptable_panel(ax, image, patch_size, targets,
                          target_color=target_color)
    _add_block_outlines(ax, patch_size, [candidate],
                        color=candidate_color, linewidth=2.5,
                        labels=['C'])


def draw_final_panel(ax, image: np.ndarray, patch_size: int,
                     targets: list[tuple[int, int, int, int]],
                     candidate: tuple[int, int, int, int],
                     target_color: str = '#c0392b',
                     candidate_color: str = '#2980b9',
                     final_rgba: tuple[float, float, float, float] = (
                         0.16, 0.50, 0.73, 0.55)):
    """Panel (d): dimmed image + solid blue tint on the surviving context patches."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    grid_h = image.shape[0] // patch_size
    grid_w = image.shape[1] // patch_size
    target_mask = _compute_union_mask(grid_h, grid_w, targets)
    candidate_mask = _compute_union_mask(grid_h, grid_w, [candidate])
    final_mask = candidate_mask & ~target_mask

    _add_per_patch_overlay(ax, patch_size, final_mask, final_rgba)

    _add_block_outlines(ax, patch_size, targets,
                        color=target_color,
                        labels=[f'T{i + 1}' for i in range(len(targets))])
    _add_block_outlines(ax, patch_size, [candidate],
                        color=candidate_color, linewidth=2.0,
                        labels=['C'])
    ax.set_axis_off()


def localized_noise_dropout_labels(turkish: bool):
    """Return panel labels for the noise-dropout figure (Mechanic 3.5)."""
    if turkish:
        return {
            'block_title_fmt': 'Blok şekli: (y, g) = ({h}, {w})',
            'sampled': '(a) Örneklenen blok',
            'noise_field': '(b) Renkli gürültü alanı',
            'thresholded_fmt': '(c) Gürültü eşikleme sonrası ({pct:.0f}% düşürüldü)',
            'cbar_label': 'Gürültü değeri',
            'cbar_low': 'düşük',
            'cbar_high': 'yüksek',
            'cbar_dropped': 'düşürüldü',
            'cbar_kept': 'tutuldu',
        }
    return {
        'block_title_fmt': 'Block shape: (h, w) = ({h}, {w})',
        'sampled': '(a) Sampled block',
        'noise_field': '(b) Colored noise field',
        'thresholded_fmt': '(c) After noise thresholding ({pct:.0f}% dropped)',
        'cbar_label': 'Noise value',
        'cbar_low': 'low',
        'cbar_high': 'high',
        'cbar_dropped': 'dropped',
        'cbar_kept': 'kept',
    }


def localized_noise_dropout_colormae_labels(turkish: bool):
    """Return panel labels for the noise-dropout ColormAE-style full-grid figure."""
    if turkish:
        return {
            'image_grid': '(a) Girdi + yama tablosu',
            'noise_field': '(b) Renkli gürültü (tüm bölgeler)',
            'thresholded_fmt': '(c) En düşük gürültülü yamaların %{pct:.0f}\'i düşürüldü',
            'cbar_label': 'Gürültü değeri',
            'cbar_low': 'düşük',
            'cbar_high': 'yüksek',
            'cbar_dropped': 'düşürüldü',
            'cbar_kept': 'tutuldu',
        }
    return {
        'image_grid': '(a) Input + patch grid',
        'noise_field': '(b) Colored noise field (full image)',
        'thresholded_fmt': '(c) Lowest-noise {pct:.0f}% of patches dropped',
        'cbar_label': 'Noise value',
        'cbar_low': 'low',
        'cbar_high': 'high',
        'cbar_dropped': 'dropped',
        'cbar_kept': 'kept',
    }


def apply_noise_threshold(block_mask_2d: np.ndarray,
                          noise_grid: np.ndarray,
                          ratio: float,
                          drop_order: str = 'lowest'
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic mirror of the noise-thresholding loop in multinoise.

    Mirrors lines ~229–252 of src/masks/multinoise.py:
      1. enumerate patches inside the rectangle
      2. read their noise values
      3. sort (descending if drop_order='lowest', else ascending)
      4. keep the first (1-ratio) fraction, drop the rest

    Returns ``(kept_mask, dropped_mask)`` — 2D bool arrays of the same
    shape as *block_mask_2d*.
    """
    box_idx = np.argwhere(block_mask_2d)
    kept = np.zeros_like(block_mask_2d, dtype=bool)
    dropped = np.zeros_like(block_mask_2d, dtype=bool)
    if len(box_idx) == 0:
        return kept, dropped

    box_noise = noise_grid[box_idx[:, 0], box_idx[:, 1]]
    descending = (drop_order == 'lowest')
    order = (np.argsort(-box_noise) if descending
             else np.argsort(box_noise))

    len_keep = int(len(box_idx) * (1.0 - ratio))
    ids_keep = order[:len_keep]
    ids_drop = order[len_keep:]

    kept[box_idx[ids_keep, 0], box_idx[ids_keep, 1]] = True
    dropped[box_idx[ids_drop, 0], box_idx[ids_drop, 1]] = True
    return kept, dropped


def draw_sampled_block_outline_panel(ax, image: np.ndarray, patch_size: int,
                                     block: tuple[int, int, int, int],
                                     color: str = '#c0392b'):
    """Panel (a): dimmed image + grid + a single block outline."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)
    _add_block_outlines(ax, patch_size, [block],
                        color=color, linewidth=2.2)
    ax.set_axis_off()


def draw_noise_field_panel(ax, image: np.ndarray, patch_size: int,
                           block: tuple[int, int, int, int] | None,
                           noise_grid: np.ndarray,
                           block_color: str = '#c0392b',
                           cmap: str = 'Greens',
                           heatmap_alpha: float = 0.55,
                           vmin: float | None = None,
                           vmax: float | None = None):
    """Panel (b): dimmed image + grid + noise heatmap + optional block outline.

    If *block* is ``None``, no rectangle is drawn (full-image ColormAE-style).

    The returned ``AxesImage`` handle uses the same cmap/vmin/vmax as the
    rendered heatmap, so a colorbar built from these values will line up
    exactly with the colors shown in the panel.
    """
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    if vmin is None:
        vmin = float(noise_grid.min())
    if vmax is None:
        vmax = float(noise_grid.max())

    noise_full = np.repeat(np.repeat(noise_grid, patch_size, axis=0),
                           patch_size, axis=1)
    im = ax.imshow(noise_full, cmap=cmap, alpha=heatmap_alpha,
                   vmin=vmin, vmax=vmax)

    if block is not None:
        _add_block_outlines(ax, patch_size, [block],
                            color=block_color, linewidth=2.2)
    ax.set_axis_off()
    return im


def draw_noise_thresholded_panel(ax, image: np.ndarray, patch_size: int,
                                 block: tuple[int, int, int, int] | None,
                                 kept_mask: np.ndarray,
                                 dropped_mask: np.ndarray,
                                 keep_color: str = '#c0392b',
                                 keep_rgba: tuple[float, float, float, float]
                                 = (0.75, 0.22, 0.17, 0.55),
                                 drop_rgba: tuple[float, float, float, float]
                                 = (0.50, 0.55, 0.55, 0.55)):
    """Panel (c): dimmed image + grid + kept/dropped tints + optional block."""
    dimmed = (image.astype(np.float32) * DIM_ALPHA).astype(np.uint8)
    _draw_grid_lines(ax, dimmed, patch_size)

    _add_per_patch_overlay(ax, patch_size, kept_mask, keep_rgba)
    _add_per_patch_overlay(ax, patch_size, dropped_mask, drop_rgba)

    if block is not None:
        _add_block_outlines(ax, patch_size, [block],
                            color=keep_color, linewidth=2.0)
    ax.set_axis_off()


def localized_noise_transform_labels(turkish: bool):
    """Return per-panel step titles for the noise-transform figure (Mechanic 3.6).

    Five titles correspond to:
      0. raw noise context (before any transform)
      1. after RandomCrop
      2. after RandomHorizontalFlip
      3. after RandomVerticalFlip
      4. after NormalizeBySliceMax
    """
    if turkish:
        return {
            'step_titles': [
                'Ham gürültü',
                '+ RandomCrop({h}\u00d7{w})',
                '+ RandomHorizontalFlip',
                '+ RandomVerticalFlip',
                '+ NormalizeBySliceMax',
            ],
            'flip_caption': 'p=0.5 ile uygulanır',
        }
    return {
        'step_titles': [
            'Raw noise',
            '+ RandomCrop({h}\u00d7{w})',
            '+ RandomHorizontalFlip',
            '+ RandomVerticalFlip',
            '+ NormalizeBySliceMax',
        ],
        'flip_caption': 'applied with p=0.5',
    }


def _draw_noise_on_image(ax, image: np.ndarray, noise_grid: np.ndarray,
                          patch_size: int, cmap: str = 'Greens',
                          alpha: float = 0.55):
    """Overlay a patch-level noise heatmap on top of *image*."""
    ax.imshow(image)
    noise_full = np.repeat(np.repeat(noise_grid, patch_size, axis=0),
                           patch_size, axis=1)
    ax.imshow(noise_full, cmap=cmap, alpha=alpha,
              vmin=float(noise_grid.min()), vmax=float(noise_grid.max()))
    ax.set_axis_off()


def _draw_transform_text_panel(ax, transform_names: list[str], header: str,
                                box_facecolor: str = '#2c3e50',
                                text_color: str = 'white'):
    """Draw a centered text box listing the transformation steps."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    bullet_lines = [f'  \u2022 {t}' for t in transform_names]
    divider = '\u2500' * 28
    text_str = f'{header}\n{divider}\n' + '\n'.join(bullet_lines)
    ax.text(0.5, 0.5, text_str,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=9, color=text_color,
            bbox=dict(facecolor=box_facecolor, alpha=0.88,
                      edgecolor='none', boxstyle='round,pad=0.7'),
            linespacing=1.8,
            fontfamily='monospace')


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
                        choices=['masks', 'patch_grid', 'block_size',
                                 'placement', 'noise_dropout',
                                 'noise_dropout_colormae',
                                 'noise_transform', 'carving', 'carving_extended'],
                        help='Which figure to generate: '
                             '"masks" (current behavior, mask panels), '
                             '"patch_grid" (Mechanic 1: image + grid overlay), '
                             '"block_size" (Mechanic 2: scale + aspect-ratio '
                             'sweep), "placement" (Mechanic 3: corner '
                             'sampling), "noise_dropout" (Mechanic 3.5: '
                             'noise-guided patch removal from a sampled '
                             'block), "noise_dropout_colormae" (full-image noise '
                             'overlay; lowest-noise patch dropout, '
                             '``--colormae_drop_ratio``), "noise_transform" '
                             '(Mechanic 3.6: '
                             'noise map before/after transformation pipeline), '
                             '"carving" (Mechanic 4: target/context '
                             'overlap removal), or "carving_extended" '
                             '(Mechanic 4 + second row for multinoise: '
                             'noise-thresholded targets and patch-accurate '
                             'complements)')
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
    parser.add_argument('--placement_block_scale', type=float, default=0.2,
                        help='Block scale used in Mechanic 3 placement figure')
    parser.add_argument('--placement_block_ar', type=float, default=1.0,
                        help='Block aspect-ratio used in Mechanic 3 figure')
    parser.add_argument('--n_placements', type=int, default=4,
                        help='Number of uniform-corner samples to overlay in '
                             'the Mechanic 3 multi-sample panel')
    parser.add_argument('--noise_block_scale', type=float, default=0.30,
                        help='Block scale used in Mechanic 3.5 noise-dropout '
                             'figure')
    parser.add_argument('--noise_block_ar', type=float, default=1.0,
                        help='Block aspect-ratio used in Mechanic 3.5 figure')
    parser.add_argument('--colormae_drop_ratio', type=float, default=0.75,
                        help='Fraction of patches dropped (lowest noise first) '
                             'for figure=noise_dropout_colormae')
    parser.add_argument('--carving_pred_scale', type=float, default=0.10,
                        help='Per-target scale used for the Mechanic 4 '
                             'figure (kept smaller than training defaults '
                             'to make individual targets visible)')
    parser.add_argument('--carving_enc_scale', type=float, default=0.55,
                        help='Context-block scale used for the Mechanic 4 '
                             'figure (kept smaller than training defaults '
                             'so candidate/target overlap is visible)')
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

    # -- Mechanic 3: where the block goes (corner sampling) ----------------
    if args.figure == 'placement':
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        grid_h = grid_w = args.input_size // ps
        pl_labels = localized_placement_labels(args.turkish)

        block_h, block_w = compute_block_hw(
            grid_h, grid_w,
            args.placement_block_scale, args.placement_block_ar)

        g = torch.Generator()
        g.manual_seed(args.seed)
        placements = sample_placements(
            grid_h, grid_w, block_h, block_w,
            n=args.n_placements, generator=g)

        single_top, single_left = placements[0]
        valid_h = max(grid_h - block_h, 0)
        valid_w = max(grid_w - block_w, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

        draw_allowed_region_panel(axes[0], image, ps, block_h, block_w)
        axes[0].set_title(pl_labels['region_title'], fontsize=11, pad=4)
        axes[0].text(
            0.5, -0.10,
            pl_labels['region_caption_fmt'].format(
                a=valid_h, b=valid_w, n=valid_h * valid_w),
            transform=axes[0].transAxes, ha='center', va='top',
            fontsize=9, color='#2c3e50')

        draw_single_placement_panel(
            axes[1], image, ps, block_h, block_w,
            top=single_top, left=single_left)
        axes[1].set_title(pl_labels['single_title'], fontsize=11, pad=4)
        axes[1].text(
            0.5, -0.10,
            pl_labels['single_caption_fmt'].format(
                r=single_top, c=single_left),
            transform=axes[1].transAxes, ha='center', va='top',
            fontsize=9, color='#2c3e50')

        draw_multi_placement_panel(
            axes[2], image, ps, block_h, block_w, placements)
        axes[2].set_title(
            pl_labels['multi_title_fmt'].format(n=args.n_placements),
            fontsize=11, pad=4)

        fig.suptitle(
            pl_labels['block_title_fmt'].format(h=block_h, w=block_w),
            fontsize=12, fontweight='bold', color='#2c3e50', y=0.99)

        plt.subplots_adjust(wspace=0.06, left=0.02, right=0.99,
                            top=0.86, bottom=0.10)

        for ext in ('png', 'pdf'):
            path = out_dir / f'mechanic3_placement.{ext}'
            fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {path.resolve()}')
        plt.close(fig)
        return

    # -- Mechanic 3.5: noise-guided patch removal --------------------------
    if args.figure == 'noise_dropout':
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        grid_h = grid_w = args.input_size // ps
        nd_labels = localized_noise_dropout_labels(args.turkish)

        block_h, block_w = compute_block_hw(
            grid_h, grid_w,
            args.noise_block_scale, args.noise_block_ar)

        torch.manual_seed(args.seed)
        g = torch.Generator()
        g.manual_seed(args.seed)
        placements = sample_placements(
            grid_h, grid_w, block_h, block_w, 1, g)
        block_top, block_left = placements[0]
        block = (block_top, block_left, block_h, block_w)

        coll = MultinoiseCollator(
            input_size=(args.input_size, args.input_size),
            patch_size=ps,
            enc_mask_scale=tuple(args.enc_mask_scale),
            pred_mask_scale=tuple(args.pred_mask_scale),
            aspect_ratio=tuple(args.aspect_ratio),
            nenc=1, npred=args.npred, min_keep=args.min_keep,
            allow_overlap=False,
            color_noise_path=args.noise_path,
            color_mask_ratio=args.color_mask_ratio,
        )
        noise_grid = coll._extract_noise_windows(1)[0].cpu().numpy()

        block_mask = np.zeros((grid_h, grid_w), dtype=bool)
        block_mask[block_top:block_top + block_h,
                   block_left:block_left + block_w] = True
        kept, dropped = apply_noise_threshold(
            block_mask, noise_grid,
            ratio=args.color_mask_ratio, drop_order='lowest')

        fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))

        draw_sampled_block_outline_panel(axes[0], image, ps, block)
        axes[0].set_title(nd_labels['sampled'], fontsize=11, pad=4)

        noise_im = draw_noise_field_panel(
            axes[1], image, ps, block, noise_grid)
        axes[1].set_title(nd_labels['noise_field'], fontsize=11, pad=4)

        draw_noise_thresholded_panel(axes[2], image, ps, block, kept, dropped)
        axes[2].set_title(
            nd_labels['thresholded_fmt'].format(
                pct=args.color_mask_ratio * 100),
            fontsize=11, pad=4)

        fig.suptitle(
            nd_labels['block_title_fmt'].format(h=block_h, w=block_w),
            fontsize=12, fontweight='bold', color='#2c3e50', y=0.99)

        plt.subplots_adjust(wspace=0.06, left=0.02, right=0.99,
                            top=0.88, bottom=0.24)

        cbar_ax = fig.add_axes([0.32, 0.13, 0.36, 0.035])
        sm = plt.cm.ScalarMappable(
            cmap=noise_im.get_cmap(), norm=noise_im.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([noise_im.norm.vmin, noise_im.norm.vmax])
        cbar.set_ticklabels([
            f"{nd_labels['cbar_low']}\n({nd_labels['cbar_dropped']})",
            f"{nd_labels['cbar_high']}\n({nd_labels['cbar_kept']})",
        ])
        for tick_label, color in zip(cbar.ax.get_xticklabels(),
                                     ['#7f8c8d', '#c0392b']):
            tick_label.set_color(color)
            tick_label.set_fontweight('bold')
            tick_label.set_fontsize(9)
        cbar.set_label(nd_labels['cbar_label'], fontsize=9, labelpad=4)

        for ext in ('png', 'pdf'):
            path = out_dir / f'mechanic3_5_noise_dropout.{ext}'
            fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {path.resolve()}')
        plt.close(fig)
        return

    # -- Noise-dropout ColormAE: full-image noise + global dropout ----------
    if args.figure == 'noise_dropout_colormae':
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        grid_h = grid_w = args.input_size // ps
        ndcm_labels = localized_noise_dropout_colormae_labels(args.turkish)

        ratio = args.colormae_drop_ratio
        if not 0.0 < ratio < 1.0:
            raise ValueError('colormae_drop_ratio must be in (0, 1)')

        torch.manual_seed(args.seed)
        coll = MultinoiseCollator(
            input_size=(args.input_size, args.input_size),
            patch_size=ps,
            enc_mask_scale=tuple(args.enc_mask_scale),
            pred_mask_scale=tuple(args.pred_mask_scale),
            aspect_ratio=tuple(args.aspect_ratio),
            nenc=1, npred=args.npred, min_keep=args.min_keep,
            allow_overlap=False,
            color_noise_path=args.noise_path,
            color_mask_ratio=args.color_mask_ratio,
        )
        noise_grid = coll._extract_noise_windows(1)[0].cpu().numpy()

        full_mask = np.ones((grid_h, grid_w), dtype=bool)
        kept, dropped = apply_noise_threshold(
            full_mask, noise_grid, ratio=ratio, drop_order='lowest')

        fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))

        _draw_grid_lines(axes[0], image, ps)
        axes[0].set_axis_off()
        axes[0].set_title(ndcm_labels['image_grid'], fontsize=11, pad=4)

        noise_im = draw_noise_field_panel(
            axes[1], image, ps, block=None, noise_grid=noise_grid)
        axes[1].set_title(ndcm_labels['noise_field'], fontsize=11, pad=4)

        draw_noise_thresholded_panel(
            axes[2], image, ps, block=None, kept_mask=kept, dropped_mask=dropped)
        axes[2].set_title(
            ndcm_labels['thresholded_fmt'].format(pct=ratio * 100),
            fontsize=11, pad=4)

        plt.subplots_adjust(wspace=0.06, left=0.02, right=0.99,
                            top=0.88, bottom=0.24)

        cbar_ax = fig.add_axes([0.32, 0.13, 0.36, 0.035])
        sm = plt.cm.ScalarMappable(
            cmap=noise_im.get_cmap(), norm=noise_im.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([noise_im.norm.vmin, noise_im.norm.vmax])
        cbar.set_ticklabels([
            f"{ndcm_labels['cbar_low']}\n({ndcm_labels['cbar_dropped']})",
            f"{ndcm_labels['cbar_high']}\n({ndcm_labels['cbar_kept']})",
        ])
        for tick_label, color in zip(cbar.ax.get_xticklabels(),
                                     ['#7f8c8d', '#c0392b']):
            tick_label.set_color(color)
            tick_label.set_fontweight('bold')
            tick_label.set_fontsize(9)
        cbar.set_label(ndcm_labels['cbar_label'], fontsize=9, labelpad=4)

        for ext in ('png', 'pdf'):
            path = out_dir / f'noise_dropout_colormae.{ext}'
            fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {path.resolve()}')
        plt.close(fig)
        return

    # -- Mechanic 3.6: noise map transformation pipeline -------------------
    if args.figure == 'noise_transform':
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        grid_h = grid_w = args.input_size // ps
        nt_labels = localized_noise_transform_labels(args.turkish)

        # Load raw noise patterns  [L, M, N]  (M, N >> grid_h)
        raw_data = np.load(args.noise_path)
        raw_tensor = torch.from_numpy(raw_data[raw_data.files[0]]).float()
        L, M, N = raw_tensor.shape

        # Pick a noise slice index (isolated generator, doesn't affect global RNG)
        g = torch.Generator()
        g.manual_seed(args.seed)
        idx = int(torch.randint(0, L, (1,), generator=g).item())

        # Determine the RandomCrop window.  get_params consumes the same random
        # numbers as the first T.RandomCrop call below (same seed), so the red
        # box on the "before" panel aligns exactly with what was actually cropped.
        torch.manual_seed(args.seed)
        top_crop, left_crop, _, _ = T.RandomCrop.get_params(
            raw_tensor, output_size=(grid_h, grid_w))

        # Context window for the "before" panel: 5× crop size around the crop
        # center so the red box occupies ~20 % of the panel width.
        ctx_scale = 5
        ctx_h = min(grid_h * ctx_scale, M)
        ctx_w = min(grid_w * ctx_scale, N)
        crop_cy = top_crop  + grid_h // 2
        crop_cx = left_crop + grid_w // 2
        ctx_top  = int(np.clip(crop_cy - ctx_h // 2, 0, M - ctx_h))
        ctx_left = int(np.clip(crop_cx - ctx_w // 2, 0, N - ctx_w))
        noise_context = raw_tensor[idx,
                                   ctx_top:ctx_top  + ctx_h,
                                   ctx_left:ctx_left + ctx_w].numpy()
        box_top  = top_crop  - ctx_top
        box_left = left_crop - ctx_left

        # Apply each transform in sequence, saving the intermediate noise slice
        # at each step.  Resetting to the same seed means the RandomCrop window
        # matches the one computed by get_params above.
        # The flips are forced (not random) so the figure always shows their
        # visual effect; the p=0.5 caption below each panel conveys the true
        # probability used during training.
        torch.manual_seed(args.seed)
        after_crop  = T.RandomCrop(grid_h)(raw_tensor)  # [L, h, w] — random, seeded
        after_hflip = after_crop.flip(-1)                # [L, h, w] — forced hflip
        after_vflip = after_hflip.flip(-2)               # [L, h, w] — forced vflip
        after_norm  = NormalizeBySliceMax()(after_vflip) # [L, h, w] — deterministic

        step_noise = [
            after_crop[idx].numpy(),
            after_hflip[idx].numpy(),
            after_vflip[idx].numpy(),
            after_norm[idx].numpy(),
        ]

        # Build per-panel step titles (fill in grid dimensions)
        step_titles = [
            t.format(h=grid_h, w=grid_w)
            for t in nt_labels['step_titles']
        ]

        # Five-panel figure ------------------------------------------------
        n_panels = 5
        fig = plt.figure(figsize=(20, 5.4))
        gs  = fig.add_gridspec(1, n_panels, wspace=0.12)
        axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

        # Panel 0: raw noise context excerpt with the crop window outlined
        axes[0].imshow(noise_context, cmap='Greens',
                       vmin=float(noise_context.min()),
                       vmax=float(noise_context.max()),
                       aspect='equal')
        axes[0].add_patch(mpatches.Rectangle(
            (box_left - 0.5, box_top - 0.5), grid_w, grid_h,
            fill=False, edgecolor='#c0392b', linewidth=2.0, zorder=3))
        axes[0].set_axis_off()
        axes[0].set_title(
            f"{step_titles[0]}\n(full size: {M}\u00d7{N})",
            fontsize=11, pad=4)

        # Panels 1–4: image + noise overlay after each cumulative transform
        for i, noise in enumerate(step_noise):
            _draw_noise_on_image(axes[i + 1], image, noise, ps)
            axes[i + 1].set_title(step_titles[i + 1], fontsize=11, pad=4)

        # Probability caption below the two flip panels (axes[2] and axes[3])
        for ax_flip in (axes[2], axes[3]):
            ax_flip.text(0.5, -0.08, nt_labels['flip_caption'],
                         transform=ax_flip.transAxes,
                         ha='center', va='top', fontsize=9,
                         color='#7f8c8d', style='italic')

        # Arrows between every adjacent pair of panels
        arrow_kw = dict(arrowstyle='->', color='#2c3e50',
                        lw=2.0, mutation_scale=20)
        for axA, axB in zip(axes[:-1], axes[1:]):
            con = mpatches.ConnectionPatch(
                xyA=(1.0, 0.5), coordsA='axes fraction', axesA=axA,
                xyB=(0.0, 0.5), coordsB='axes fraction', axesB=axB,
                **arrow_kw, zorder=4)
            fig.add_artist(con)

        plt.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.12)

        for ext in ('png', 'pdf'):
            path = out_dir / f'mechanic3_6_noise_transform.{ext}'
            fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
            print(f'Saved {path.resolve()}')
        plt.close(fig)
        return

    # -- Mechanic 4: the carving trick (target/context overlap removal) ----
    if args.figure in ('carving', 'carving_extended'):
        img_path = args.image_path[0]
        image = load_image(img_path, args.input_size)
        grid_h = grid_w = args.input_size // ps
        cv_labels = localized_carving_labels(args.turkish)

        target_h, target_w = compute_block_hw(
            grid_h, grid_w, args.carving_pred_scale, 1.0)
        context_h, context_w = compute_block_hw(
            grid_h, grid_w, args.carving_enc_scale, 1.0)

        g = torch.Generator()
        g.manual_seed(args.seed)
        target_corners = sample_placements(
            grid_h, grid_w, target_h, target_w, args.npred, g)
        targets = [(t, l, target_h, target_w) for (t, l) in target_corners]

        context_corner = sample_placements(
            grid_h, grid_w, context_h, context_w, 1, g)[0]
        candidate = (context_corner[0], context_corner[1],
                     context_h, context_w)

        if args.figure == 'carving':
            fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))

            draw_targets_only_panel(axes[0], image, ps, targets)
            axes[0].set_title(cv_labels['targets'], fontsize=11, pad=4)

            draw_acceptable_panel(axes[1], image, ps, targets)
            axes[1].set_title(cv_labels['acceptable'], fontsize=11, pad=4)

            draw_candidate_panel(axes[2], image, ps, targets, candidate)
            axes[2].set_title(cv_labels['candidate'], fontsize=11, pad=4)

            draw_final_panel(axes[3], image, ps, targets, candidate)
            axes[3].set_title(cv_labels['final'], fontsize=11, pad=4)

            fig.text(0.5, 0.03, cv_labels['caption'],
                     ha='center', fontsize=10, style='italic', color='#34495e')

            plt.subplots_adjust(wspace=0.06, left=0.02, right=0.99,
                                top=0.92, bottom=0.10)

            for ext in ('png', 'pdf'):
                path = out_dir / f'mechanic4_carving.{ext}'
                fig.savefig(str(path), dpi=args.dpi, bbox_inches='tight')
                print(f'Saved {path.resolve()}')
            plt.close(fig)
            return

        ext_labels = localized_carving_extended_labels(args.turkish)
        coll = MultinoiseCollator(
            input_size=(args.input_size, args.input_size),
            patch_size=ps,
            enc_mask_scale=tuple(args.enc_mask_scale),
            pred_mask_scale=tuple(args.pred_mask_scale),
            aspect_ratio=tuple(args.aspect_ratio),
            nenc=1, npred=args.npred, min_keep=args.min_keep,
            allow_overlap=False,
            color_noise_path=args.noise_path,
            color_mask_ratio=args.color_mask_ratio,
            enc_drop_order=args.enc_drop_order,
            pred_drop_order=args.pred_drop_order,
        )
        noise_grid = coll._extract_noise_windows(1)[0].cpu().numpy()
        target_kepts, _acceptable, _cand_noise, final_carved = (
            compute_multinoise_carving_state(
                grid_h, grid_w, targets, candidate, noise_grid,
                args.color_mask_ratio,
                pred_drop_order=args.pred_drop_order,
                enc_drop_order=args.enc_drop_order))

        fig, axes = plt.subplots(2, 4, figsize=(15, 8.6))

        draw_targets_only_panel(axes[0, 0], image, ps, targets)
        axes[0, 0].set_title(cv_labels['targets'], fontsize=11, pad=4)

        draw_acceptable_panel(axes[0, 1], image, ps, targets)
        axes[0, 1].set_title(cv_labels['acceptable'], fontsize=11, pad=4)

        draw_candidate_panel(axes[0, 2], image, ps, targets, candidate)
        axes[0, 2].set_title(cv_labels['candidate'], fontsize=11, pad=4)

        draw_final_panel(axes[0, 3], image, ps, targets, candidate)
        axes[0, 3].set_title(cv_labels['final'], fontsize=11, pad=4)

        draw_multinoise_targets_kept_panel(
            axes[1, 0], image, ps, target_kepts, targets)
        axes[1, 0].set_title(cv_labels['targets'], fontsize=11, pad=4)

        draw_multinoise_acceptable_kept_panel(
            axes[1, 1], image, ps, target_kepts, targets)
        axes[1, 1].set_title(cv_labels['acceptable'], fontsize=11, pad=4)

        draw_multinoise_candidate_kept_panel(
            axes[1, 2], image, ps, target_kepts, targets, candidate)
        axes[1, 2].set_title(cv_labels['candidate'], fontsize=11, pad=4)

        draw_multinoise_final_kept_panel(
            axes[1, 3], image, ps, targets, candidate, final_carved)
        axes[1, 3].set_title(cv_labels['final'], fontsize=11, pad=4)

        fig.text(0.5, 0.04, cv_labels['caption'],
                 ha='center', fontsize=9, style='italic', color='#34495e')

        side_kw = dict(
            rotation=90, va='center', ha='center',
            fontsize=20, fontweight='bold', color='#2c3e50')
        fig.text(0.035, 0.705, ext_labels['side_multiblock'], **side_kw)
        fig.text(0.035, 0.315, ext_labels['side_multinoise'], **side_kw)

        plt.subplots_adjust(wspace=0.06, hspace=0.28,
                            left=0.09, right=0.99, top=0.90, bottom=0.11)

        for ext in ('png', 'pdf'):
            path = out_dir / f'mechanic4_carving_extended.{ext}'
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
