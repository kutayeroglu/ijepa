"""Distributed multi-GPU linear probe training for I-JEPA ViT encoders.

Follows the same mp.Process spawning pattern as main.py / train.py.
Each process gets one GPU via CUDA_VISIBLE_DEVICES, initialises NCCL,
and runs DDP training.  Single-GPU (--devices cuda:0) works without DDP.
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import src.models.vision_transformer as vit
from src.datasets.singleGPU_imagenet1k import ImageNetFlatValDataset
from src.models.vit_linear_probe import LinearProbeModel
from src.models.vision_transformer import VIT_EMBED_DIMS
from src.utils.distributed import init_distributed
from src.utils.linprobe_trainer import train_linear_probe
from src.utils.run_tracking import (
    build_run_id,
    checkpoint_stem,
    ensure_dir,
    get_runtime_context,
    timestamp_utc,
    write_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Distributed linear probe training")
    p.add_argument("--dataset_dir", type=str, default="~/datasets",
                   help="Base directory for datasets")
    p.add_argument("--val_dir", type=str, default=None,
                   help="Explicit validation directory (overrides dataset_dir/…/val)")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Per-GPU batch size (effective = batch_size × num_gpus)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=0.00625,
                   help="Learning rate (linearly scaled from 0.05 for BS=2048)")
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--val_labels_file", type=str, default=None,
                   help="Ground-truth labels file for flat validation directory")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--model_name", type=str, default="vit_huge",
                   choices=["vit_tiny", "vit_small", "vit_base",
                            "vit_large", "vit_huge", "vit_giant"])
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--encoder_key", type=str, default="target_encoder",
                   choices=["encoder", "target_encoder"])
    p.add_argument("--output_root", type=str,
                   default=os.environ.get(
                       "IJEPA_LINPROBE_ROOT",
                       os.path.join("~", "outputs", "ijepa", "linprobe")))
    p.add_argument("--outputs_dir", type=str, default=None,
                   help="Explicit output directory (overrides output_root/…/runs/run_id)")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--devices", type=str, nargs="+", default=["cuda:0"],
                   help="GPU devices (e.g. cuda:0 cuda:1 cuda:2 cuda:3)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_model_path(script_dir, maybe_path):
    if maybe_path:
        return os.path.abspath(os.path.expanduser(maybe_path))
    return os.path.join(script_dir, "pretrained_models", "IN1K-vit.h.14-300e.pth.tar")


def _build_outputs_dir(output_root, run_id, model_path):
    output_root = os.path.abspath(os.path.expanduser(output_root))
    tag = checkpoint_stem(model_path)
    experiment_dir = ensure_dir(os.path.join(output_root, tag))
    runs_root = ensure_dir(os.path.join(experiment_dir, "runs"))
    resolved_id = run_id or build_run_id(f"lprobe_{tag}")
    outputs_dir = ensure_dir(os.path.join(runs_root, resolved_id))
    return output_root, tag, experiment_dir, runs_root, resolved_id, outputs_dir


# ---------------------------------------------------------------------------
# Distributed data loaders
# ---------------------------------------------------------------------------

def _make_dataloaders(data_dir, val_dir, batch_size, num_workers,
                      world_size, rank, val_labels_file=None):
    """Build train/val loaders.  Uses DistributedSampler when world_size > 1."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform)
    logger.info("Train dataset: %d classes, %d samples",
                len(train_dataset.classes), len(train_dataset))

    resolved_val_dir = val_dir if val_dir is not None else os.path.join(data_dir, "val")
    if not os.path.isdir(resolved_val_dir):
        raise FileNotFoundError(f"Validation directory not found: {resolved_val_dir}")

    has_subdirs = any(
        os.path.isdir(os.path.join(resolved_val_dir, d))
        for d in os.listdir(resolved_val_dir)
    )
    if has_subdirs:
        val_dataset = datasets.ImageFolder(resolved_val_dir, transform=val_transform)
        logger.info("Val dataset (ImageFolder): %d classes, %d samples",
                     len(val_dataset.classes), len(val_dataset))
    else:
        if val_labels_file is None:
            raise ValueError(
                "val_labels_file required for flat validation directory")
        if not os.path.exists(val_labels_file):
            raise FileNotFoundError(f"Val labels file not found: {val_labels_file}")
        val_dataset = ImageNetFlatValDataset(
            resolved_val_dir, val_labels_file, transform=val_transform)
        logger.info("Val dataset (flat): %d samples", len(val_dataset))

    distributed = world_size > 1
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=distributed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Per-process entry point
# ---------------------------------------------------------------------------

def process_main(rank, args, world_size, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
        force=True,
    )

    distributed = world_size > 1
    if distributed:
        world_size, rank = init_distributed(
            rank_and_world_size=(rank, world_size))
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Failed to initialise distributed process group")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    logger.info("Initialised rank %d / %d  (distributed=%s)",
                rank, world_size, distributed)

    # ---- paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = _default_model_path(script_dir, args.model_path)
    source_tag = checkpoint_stem(model_path)

    if args.outputs_dir:
        outputs_dir = ensure_dir(
            os.path.abspath(os.path.expanduser(args.outputs_dir)))
        run_id = args.run_id or os.path.basename(outputs_dir.rstrip(os.sep))
        output_root = os.path.dirname(outputs_dir)
        experiment_dir = output_root
        runs_root = output_root
    else:
        (output_root, source_tag, experiment_dir,
         runs_root, run_id, outputs_dir) = _build_outputs_dir(
            args.output_root, args.run_id, model_path)

    log_file_path = os.path.join(outputs_dir, "training_log")
    if rank == 0:
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s:%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)

    logger.info("All arguments: %s", vars(args))

    # ---- dataset directory ----
    dataset_dir = os.path.expanduser(args.dataset_dir)
    val_dir = os.path.expanduser(args.val_dir) if args.val_dir else None
    in1k_dir = os.path.join(dataset_dir, "in1k")
    if not os.path.exists(in1k_dir):
        in1k_dir = dataset_dir

    logger.info("Data root: %s", in1k_dir)
    if val_dir:
        logger.info("Validation directory override: %s", val_dir)

    # ---- manifest ----
    manifest_path = os.path.join(outputs_dir, "run_manifest.json")
    run_started_at = timestamp_utc()
    runtime_ctx = get_runtime_context()

    def write_manifest(status, extra=None):
        if rank != 0:
            return
        payload = {
            "task_type": "linear_probe",
            "distributed": distributed,
            "world_size": world_size,
            "status": status,
            "run_id": run_id,
            "output_root": output_root,
            "experiment_dir": experiment_dir,
            "runs_root": runs_root,
            "output_dir": outputs_dir,
            "log_file_path": log_file_path,
            "source_checkpoint_path": model_path,
            "source_checkpoint_tag": source_tag,
            "model_name": args.model_name,
            "patch_size": args.patch_size,
            "encoder_key": args.encoder_key,
            "dataset_dir": dataset_dir,
            "val_dir": val_dir,
            "batch_size_per_gpu": args.batch_size,
            "effective_batch_size": args.batch_size * world_size,
            "num_workers": args.num_workers,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "started_at": run_started_at,
            "arguments": vars(args),
        }
        payload.update(runtime_ctx)
        if extra:
            payload.update(extra)
        write_json(manifest_path,
                   {k: v for k, v in payload.items() if v not in (None, "")})

    write_manifest("running")

    # ---- load encoder ----
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        logger.info("Loaded checkpoint: %s", model_path)
    except Exception as exc:
        write_manifest("failed",
                       {"completed_at": timestamp_utc(), "error": str(exc)})
        logger.exception("Failed to load checkpoint")
        raise

    encoder_state = checkpoint.get(args.encoder_key)
    if encoder_state is None:
        msg = (f"Key '{args.encoder_key}' not in checkpoint. "
               f"Available: {list(checkpoint.keys())}")
        write_manifest("failed",
                       {"completed_at": timestamp_utc(), "error": msg})
        raise KeyError(msg)

    cleaned = OrderedDict()
    for k, v in encoder_state.items():
        cleaned[k.replace("module.", "")] = v

    encoder = vit.__dict__[args.model_name](
        patch_size=args.patch_size, img_size=[args.img_size])
    encoder.load_state_dict(cleaned)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    embed_dim = VIT_EMBED_DIMS[args.model_name]
    model = LinearProbeModel(
        encoder=encoder, embed_dim=embed_dim, num_classes=1000)

    logger.info(
        "Model: %s  patch=%d  img=%d  embed=%d  classes=%d  key=%s",
        args.model_name, args.patch_size, args.img_size,
        model.classifier.in_features, model.classifier.out_features,
        args.encoder_key,
    )

    # ---- move to device, wrap DDP ----
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(model)

    # ---- data loaders ----
    train_loader, val_loader, train_sampler = _make_dataloaders(
        in1k_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        val_labels_file=args.val_labels_file,
    )

    # ---- train ----
    try:
        result = train_linear_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            outputs_dir=outputs_dir,
            rank=rank,
            world_size=world_size,
            train_sampler=train_sampler,
        )
    except Exception as exc:
        write_manifest("failed",
                       {"completed_at": timestamp_utc(), "error": str(exc)})
        raise

    write_manifest("completed", {
        "completed_at": timestamp_utc(),
        "best_val_acc": result["best_acc"],
        "best_checkpoint_path": result["best_checkpoint_path"],
        "metrics_path": result["metrics_path"],
        "plot_path": result["plot_path"],
    })

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    num_gpus = len(args.devices)

    if num_gpus == 1:
        process_main(0, args, 1, args.devices)
    else:
        mp.set_start_method("spawn")
        processes = []
        for gpu_rank in range(num_gpus):
            p = mp.Process(
                target=process_main,
                args=(gpu_rank, args, num_gpus, args.devices))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
