import logging
import os
from datetime import datetime
from collections import OrderedDict

import torch

from src.models.vision_transformer import vit_huge
from src.models.vit_linear_probe import LinearProbeModel
from src.datasets.singleGPU_imagenet1k import get_imagenet_dataloaders
from src.utils.linprobe_trainer import train_linear_probe


# --- SETUP OUTPUT DIRECTORY AND LOGGING ---
project_name = "ijepa"
run_name = f"linprobe_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

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
    # Params
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "pretrained_models")
    model_file_name = "IN1K-vit.h.14-300e.pth.tar"
    model_path = os.path.join(model_dir, model_file_name)

    dataset_dir = os.path.join("~/datasets")
    in1k_dir = os.path.join(dataset_dir, "in1k")

    # Read encoder
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        logger.info(f"Loaded model successfully from: {model_path}")
    except Exception as e:
        logger.exception(f"Error loading the model from {model_path}: {e}")

    # NOTE: which one to use from checkpoint: encoder or target_encoder?? why?
    encoder = checkpoint["target_encoder"]

    # Clean state dict keys
    new_state_dict = OrderedDict()
    for k, v in encoder.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    # Load state dict
    vit_h_encoder = vit_huge(
        patch_size=14,
        img_size=[224],
    )
    vit_h_encoder.load_state_dict(new_state_dict)

    # Freeze weights
    for param in vit_h_encoder.parameters():
        param.requires_grad = False
    vit_h_encoder.eval()

    # Create model with linear head
    model = LinearProbeModel(
        encoder=vit_h_encoder,
        embed_dim=1280,
        num_classes=1000,
    )

    logger.info("Created model with linear head")

    # Get dataloaders
    train_loader, val_loader = get_imagenet_dataloaders(
        in1k_dir,
        batch_size=32,  # NOTE: Adjust based on GPU memory
        num_workers=4,  # NOTE: num_GPU * 4 or os.cpu_count()
        train_frac=0.0025,
        val_frac=0.1,
    )

    # TODO: get params with argparse
    # Train linear head on in1k-trainset
    trained_model = train_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        learning_rate=0.1,
        device="cuda",
        outputs_dir=outputs_dir,
    )

    # TODO: Evaluate on in1k-valset
