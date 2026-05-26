"""English / Turkish strings for mask visualization figures."""

EN_MASK_TYPE_LABELS = {
    'multinoise': 'multi-noise',
    'multiblock': 'multi-block',
}

TR_MASK_TYPE_LABELS = {
    'multinoise': 'Çoklu Gürültü',
    'multiblock': 'Çoklu Blok',
}


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


def patch_grid_annotation_text(patch_size: int, turkish: bool) -> str:
    """Annotation beside the highlighted patch in the patch-grid figure."""
    if turkish:
        return f'1 yama\n= {patch_size}×{patch_size} piksel'
    return f'1 patch\n= {patch_size}×{patch_size} px'


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
            'sample_title': ('Örneklenen yapılandırma\n'
                               r'$n_{{\mathrm{{pred}}}}$ = {n} hedef'),
        }
    return {
        'title_fmt': ('scale = {s:.2f}, ar = {ar:.2f}\n'
                      '(h, w) = ({h}, {w}), {n} patches'),
        'row_scale': 'Scale controls area',
        'row_ar': 'Aspect ratio controls shape',
        'caption': ('Target block: aspect_ratio ∈ (0.75, 1.5);  '
                    'context block: aspect_ratio = 1.0 (square).'),
        'sample_title': ('Sampled configuration\n'
                           r'$n_{{\mathrm{{pred}}}}$ = {n} targets'),
    }


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


def localized_carving_labels(turkish: bool):
    """Return panel labels for the carving figure (Mechanic 4)."""
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


def localized_mask_type_label(mask_type: str, turkish: bool) -> str:
    """Return display label for the selected mask type."""
    if turkish:
        return TR_MASK_TYPE_LABELS.get(mask_type, mask_type)
    return EN_MASK_TYPE_LABELS.get(mask_type, mask_type)
