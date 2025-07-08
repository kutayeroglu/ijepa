import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler


def get_imagenet_dataloaders(
    data_dir,
    batch_size=1024,
    num_workers=8,
    train_frac: float = None,
    val_frac: float = None,
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

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
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
