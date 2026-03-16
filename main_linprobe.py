"""Linear probe training for I-JEPA ViT encoders.

Usage examples:

    # ViT-Huge (default)
    python main_linprobe.py \
        --model_path path/to/checkpoint.pth.tar

    # ViT-Base
    python main_linprobe.py \
        --model_name vit_base \
        --patch_size 16 \
        --model_path path/to/vit_base.pth.tar

    # ViT-Small
    python main_linprobe.py \
        --model_name vit_small \
        --patch_size 16 \
        --model_path path/to/vit_small.pth.tar

    # With additional options
    python main_linprobe.py \
        --model_name vit_base \
        --patch_size 16 \
        --model_path path/to/checkpoint.pth.tar \
        --dataset_dir ~/datasets \
        --batch_size 128
"""
import argparse
import logging
import os
from datetime import datetime
from collections import OrderedDict

import torch

import src.models.vision_transformer as vit
from src.models.vision_transformer import VIT_EMBED_DIMS
from src.models.vit_linear_probe import LinearProbeModel
from src.datasets.singleGPU_imagenet1k import get_imagenet_dataloaders
from src.utils.linprobe_trainer import train_linear_probe


# --- SETUP OUTPUT DIRECTORY AND LOGGING ---
project_name = "ijepa"
run_name = f"lprobe_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Create unique run directory
outputs_dir = os.path.join(os.path.expanduser("~"), "outputs", project_name, run_name)
os.makedirs(outputs_dir, exist_ok=True)

# Setup logging
log_file_path = os.path.join(outputs_dir, "training_log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Linear probe training")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="~/datasets",
        help="Base directory for datasets (default: ~/datasets)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Optional explicit validation directory (default: dataset_dir/in1k/val or dataset_dir/val)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1.0,
        help="Fraction of training data to use (default: 1.0)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=1.0,
        help="Fraction of validation data to use (default: 1.0)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00625,
        help="Learning rate (default: 0.00625, linearly scaled from 0.05 for BS=2048)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Weight decay (default: 0.0005)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--val_labels_file",
        type=str,
        default=None,
        help="Path to ground truth labels file for flat validation structure (default: None)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to checkpoint file (default: uses pretrained_models/IN1K-vit.h.14-300e.pth.tar)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_huge",
        choices=["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge", "vit_giant"],
        help="ViT model architecture (default: vit_huge)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Patch size (default: 14 for vit.h.14, use 16 for vit.b/s/l/g)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--encoder_key",
        type=str,
        default="target_encoder",
        choices=["encoder", "target_encoder"],
        help="Checkpoint key for encoder weights (default: target_encoder)",
    )

    args = parser.parse_args()
    logger.info(f"All arguments: {vars(args)}")

    # Params
    script_dir = os.path.dirname(__file__)
    if args.model_path:
        model_path = os.path.expanduser(args.model_path)
    else:
        model_dir = os.path.join(script_dir, "pretrained_models")
        model_file_name = "IN1K-vit.h.14-300e.pth.tar"
        model_path = os.path.join(model_dir, model_file_name)

    dataset_dir = os.path.expanduser(args.dataset_dir)
    val_dir = os.path.expanduser(args.val_dir) if args.val_dir else None
    # Check if in1k subdirectory exists, otherwise use the dataset_dir directly
    in1k_dir = os.path.join(dataset_dir, "in1k")
    if not os.path.exists(in1k_dir):
        in1k_dir = dataset_dir

    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Using data from: {in1k_dir}")
    if val_dir:
        logger.info(f"Validation directory override: {val_dir}")

    # Read encoder
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        logger.info(f"Loaded model successfully from: {model_path}")
    except Exception as e:
        logger.exception(f"Error loading the model from {model_path}: {e}")
        raise

    encoder_state = checkpoint.get(args.encoder_key)
    if encoder_state is None:
        raise KeyError(
            f"Checkpoint missing key '{args.encoder_key}'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    # Clean state dict keys
    new_state_dict = OrderedDict()
    for k, v in encoder_state.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    # Build encoder dynamically
    model_builder = vit.__dict__[args.model_name]
    encoder = model_builder(
        patch_size=args.patch_size,
        img_size=[args.img_size],
    )
    encoder.load_state_dict(new_state_dict)

    # Freeze weights
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    embed_dim = VIT_EMBED_DIMS[args.model_name]

    # Create model with linear head
    model = LinearProbeModel(
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=1000,
    )

    logger.info(
        f"Model config: model_name={args.model_name}, "
        f"patch_size={encoder.patch_embed.patch_size}, "
        f"img_size={encoder.patch_embed.img_size}, "
        f"embed_dim={model.classifier.in_features}, "
        f"num_classes={model.classifier.out_features}, encoder_key={args.encoder_key}"
    )

    # Get dataloaders
    train_loader, val_loader = get_imagenet_dataloaders(
        in1k_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        val_dir=val_dir,
        val_labels_file=args.val_labels_file,
    )

    # Train linear head on in1k-trainset
    trained_model = train_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        outputs_dir=outputs_dir,
    )
