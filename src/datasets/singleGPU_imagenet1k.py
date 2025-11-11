import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
from PIL import Image


class ImageNetFlatValDataset(Dataset):
    """Custom dataset for ImageNet validation when images are in a flat directory."""

    def __init__(self, val_dir, val_labels_file, transform=None):
        """
        Args:
            val_dir: Directory containing validation images
            val_labels_file: Path to ground truth labels file (ILSVRC2012_validation_ground_truth.txt)
            transform: Optional transform to be applied on a sample
        """
        self.val_dir = val_dir
        self.transform = transform

        # Load image files - ImageNet validation images are .JPEG format
        self.image_files = sorted(
            [f for f in os.listdir(val_dir) if f.endswith(".JPEG")]
        )

        # Load ground truth labels
        self.labels = []
        with open(val_labels_file, "r") as f:
            for line in f:
                # Labels in ground truth file are 1-indexed
                self.labels.append(int(line.strip()) - 1)  # Convert to 0-indexed

        # Ensure we have labels for all images
        if len(self.labels) != len(self.image_files):
            raise ValueError(
                f"Mismatch: {len(self.image_files)} images but {len(self.labels)} labels. "
                f"Expected {len(self.image_files)} labels in ground truth file."
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.val_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_imagenet_dataloaders(
    data_dir,
    val_dir: Optional[str] = None,
    batch_size=1024,
    num_workers=8,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
    val_labels_file: Optional[str] = None,
):
    """Create ImageNet dataloaders with appropriate transforms"""

    # Training transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation transforms
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )

    # Handle validation dataset: check if it's in subdirectories or flat structure
    resolved_val_dir = val_dir if val_dir is not None else os.path.join(data_dir, "val")

    if not os.path.isdir(resolved_val_dir):
        raise FileNotFoundError(
            f"Validation directory not found at: {resolved_val_dir}. "
            "Provide a valid path via `val_dir`."
        )

    # Check if validation directory has subdirectories
    has_subdirs = any(
        os.path.isdir(os.path.join(resolved_val_dir, item))
        for item in os.listdir(resolved_val_dir)
        if os.path.isdir(os.path.join(resolved_val_dir, item))
    )

    if has_subdirs:
        # Use ImageFolder for subdirectory structure (local format)
        val_dataset = datasets.ImageFolder(resolved_val_dir, transform=val_transform)
        print("Using ImageFolder for validation (subdirectory structure)")
    else:
        # Use custom dataset for flat structure (HPC format)
        if val_labels_file is None:
            raise ValueError(
                "val_labels_file parameter is required when validation images are in flat structure"
            )

        if not os.path.exists(val_labels_file):
            raise FileNotFoundError(
                f"Validation images are in flat structure but cannot find ground truth labels file at: {val_labels_file}"
            )

        print(
            f"Using custom dataset for validation (flat structure) with labels from: {val_labels_file}"
        )
        val_dataset = ImageNetFlatValDataset(
            resolved_val_dir, val_labels_file, transform=val_transform
        )

    # Create samplers
    train_sampler = None
    val_sampler = None

    if train_frac is not None:
        num = len(train_dataset)
        # reproducible random split
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(num, generator=g)[: int(num * train_frac)].tolist()
        train_sampler = SubsetRandomSampler(indices)
    if val_frac is not None:
        num = len(val_dataset)
        g = torch.Generator().manual_seed(43)
        indices = torch.randperm(num, generator=g)[: int(num * val_frac)].tolist()
        val_sampler = SubsetRandomSampler(indices)

    # Create dataloaders
    # NOTE: for pin_memory = True, see: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
