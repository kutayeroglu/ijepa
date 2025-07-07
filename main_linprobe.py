import os
from collections import OrderedDict

import torch

from src.models.vision_transformer import vit_huge
from src.models.vit_linear_probe import LinearProbeModel
from src.datasets.singleGPU_imagenet1k import get_imagenet_dataloaders
from src.utils.linprobe_trainer import train_linear_probe


if __name__ == "__main__":
    # Params
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "pretrained_models")
    model_file_name = "IN1K-vit.h.14-300e.pth.tar"
    model_path = os.path.join(model_dir, model_file_name)

    dataset_dir = os.path.join(script_dir, "datasets")
    in1k_dir = os.path.join(dataset_dir, "in1k")

    # Read encoder
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")

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

    print("Model created with linear head")

    # Get dataloaders
    train_loader, val_loader = get_imagenet_dataloaders(
        in1k_dir,
        batch_size=32,  # NOTE: Adjust based on GPU memory
        num_workers=8,
    )

    # TODO: get params with argparse
    # Train linear head on in1k-trainset
    train_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.1,
        device="cuda",
    )

    # Evaluate on in1k-valset
