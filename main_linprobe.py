"""Linear probe training for I-JEPA ViT encoders."""
import argparse
import logging
import os
from collections import OrderedDict

import torch

import src.models.vision_transformer as vit
from src.datasets.singleGPU_imagenet1k import get_imagenet_dataloaders
from src.models.vit_linear_probe import LinearProbeModel
from src.models.vision_transformer import VIT_EMBED_DIMS
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


def parse_args():
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
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.environ.get(
            "IJEPA_LINPROBE_ROOT",
            os.path.join("~", "outputs", "ijepa", "linprobe"),
        ),
        help="Root directory under which linear-probe run folders are created (ignored when --outputs_dir is set).",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=None,
        help="Optional explicit output directory. When set, used directly instead of output_root/.../runs/run_id.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional explicit run ID. Defaults to a timestamped name derived from the source checkpoint.",
    )
    return parser.parse_args()


def configure_logging(outputs_dir):
    log_file_path = os.path.join(outputs_dir, "training_log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
        force=True,
    )
    return log_file_path


def default_model_path(script_dir, maybe_model_path):
    if maybe_model_path:
        return os.path.abspath(os.path.expanduser(maybe_model_path))

    model_dir = os.path.join(script_dir, "pretrained_models")
    model_file_name = "IN1K-vit.h.14-300e.pth.tar"
    return os.path.join(model_dir, model_file_name)


def build_outputs_dir(output_root, run_id, model_path):
    output_root = os.path.abspath(os.path.expanduser(output_root))
    source_checkpoint_tag = checkpoint_stem(model_path)
    experiment_dir = ensure_dir(os.path.join(output_root, source_checkpoint_tag))
    runs_root = ensure_dir(os.path.join(experiment_dir, "runs"))
    resolved_run_id = run_id or build_run_id(f"lprobe_{source_checkpoint_tag}")
    outputs_dir = ensure_dir(os.path.join(runs_root, resolved_run_id))
    return (
        output_root,
        source_checkpoint_tag,
        experiment_dir,
        runs_root,
        resolved_run_id,
        outputs_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    script_dir = os.path.dirname(__file__)
    model_path = default_model_path(script_dir, args.model_path)
    source_checkpoint_tag = checkpoint_stem(model_path)

    if args.outputs_dir:
        outputs_dir = ensure_dir(os.path.abspath(os.path.expanduser(args.outputs_dir)))
        run_id = args.run_id or os.path.basename(outputs_dir.rstrip(os.sep))
        output_root = os.path.dirname(outputs_dir)
        experiment_dir = output_root
        runs_root = output_root
    else:
        (
            output_root,
            source_checkpoint_tag,
            experiment_dir,
            runs_root,
            run_id,
            outputs_dir,
        ) = build_outputs_dir(
            args.output_root,
            args.run_id,
            model_path,
        )
    log_file_path = configure_logging(outputs_dir)
    logger.info("All arguments: %s", vars(args))

    dataset_dir = os.path.expanduser(args.dataset_dir)
    val_dir = os.path.expanduser(args.val_dir) if args.val_dir else None
    in1k_dir = os.path.join(dataset_dir, "in1k")
    if not os.path.exists(in1k_dir):
        in1k_dir = dataset_dir

    logger.info("Dataset directory: %s", dataset_dir)
    logger.info("Using data from: %s", in1k_dir)
    if val_dir:
        logger.info("Validation directory override: %s", val_dir)

    manifest_path = os.path.join(outputs_dir, "run_manifest.json")
    run_started_at = timestamp_utc()
    runtime_context = get_runtime_context()

    def write_run_manifest(status, extra=None):
        payload = {
            "task_type": "linear_probe",
            "status": status,
            "run_id": run_id,
            "output_root": output_root,
            "experiment_dir": experiment_dir,
            "runs_root": runs_root,
            "output_dir": outputs_dir,
            "log_file_path": log_file_path,
            "source_checkpoint_path": model_path,
            "source_checkpoint_tag": source_checkpoint_tag,
            "model_name": args.model_name,
            "patch_size": args.patch_size,
            "encoder_key": args.encoder_key,
            "dataset_dir": dataset_dir,
            "val_dir": val_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "started_at": run_started_at,
            "arguments": vars(args),
        }
        payload.update(runtime_context)
        if extra:
            payload.update(extra)
        write_json(
            manifest_path,
            {key: value for key, value in payload.items() if value not in (None, "")},
        )

    write_run_manifest("running")

    # Read encoder
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        logger.info("Loaded model successfully from: %s", model_path)
    except Exception as error:
        write_run_manifest(
            "failed",
            {"completed_at": timestamp_utc(), "error": str(error)},
        )
        logger.exception("Error loading the model from %s: %s", model_path, error)
        raise

    encoder_state = checkpoint.get(args.encoder_key)
    if encoder_state is None:
        available_keys = list(checkpoint.keys())
        error = (
            f"Checkpoint missing key '{args.encoder_key}'. "
            f"Available keys: {available_keys}"
        )
        write_run_manifest(
            "failed",
            {"completed_at": timestamp_utc(), "error": error},
        )
        raise KeyError(error)

    # Clean state dict keys
    new_state_dict = OrderedDict()
    for k, v in encoder_state.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

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
        "Model config: model_name=%s, patch_size=%s, img_size=%s, embed_dim=%s, "
        "num_classes=%s, encoder_key=%s",
        args.model_name,
        encoder.patch_embed.patch_size,
        encoder.patch_embed.img_size,
        model.classifier.in_features,
        model.classifier.out_features,
        args.encoder_key,
    )

    train_loader, val_loader = get_imagenet_dataloaders(
        in1k_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        val_dir=val_dir,
        val_labels_file=args.val_labels_file,
    )

    try:
        training_result = train_linear_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            outputs_dir=outputs_dir,
        )
    except Exception as error:
        write_run_manifest(
            "failed",
            {"completed_at": timestamp_utc(), "error": str(error)},
        )
        raise

    write_run_manifest(
        "completed",
        {
            "completed_at": timestamp_utc(),
            "best_val_acc": training_result["best_acc"],
            "best_checkpoint_path": training_result["best_checkpoint_path"],
            "metrics_path": training_result["metrics_path"],
            "plot_path": training_result["plot_path"],
        },
    )
