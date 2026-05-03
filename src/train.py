# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.multinoise import MaskCollator as NoiseMaskCollator
from src.masks.ng_multiblock import MaskCollator as NGMBMaskCollator
from src.masks.quadrantnoise import MaskCollator as QuadrantNoiseMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.run_tracking import (
    build_run_id,
    ensure_dir,
    get_runtime_context,
    timestamp_utc,
    write_json,
)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    copy_data = args["meta"]["copy_data"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # Clear CUDA cache
        torch.cuda.empty_cache()

    # -- DATA
    use_gaussian_blur = args["data"]["use_gaussian_blur"]
    use_horizontal_flip = args["data"]["use_horizontal_flip"]
    use_color_distortion = args["data"]["use_color_distortion"]
    color_jitter = args["data"]["color_jitter_strength"]
    # --
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    image_folder = args["data"]["image_folder"]
    crop_size = args["data"]["crop_size"]
    crop_scale = args["data"]["crop_scale"]
    train_fraction = args["data"].get("train_fraction", None)  # Optional fraction of training set
    # --

    # -- MASK
    allow_overlap = args["mask"][
        "allow_overlap"
    ]  # whether to allow overlap b/w context and target blocks
    patch_size = args["mask"]["patch_size"]  # patch-size for model training
    num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
    min_keep = args["mask"]["min_keep"]  # min number of patches in context block
    enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
    num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
    pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
    aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks
    mask_type = args["mask"].get("mask_type", "multiblock")  # multiblock | multinoise | quadrantnoise | ng_multiblock
    green_noise_data_path = args["mask"].get("green_noise_data_path", None)  # path to color noise patterns
    color_mask_ratio = args["mask"].get("color_mask_ratio", 0.15)
    enc_drop_order = args["mask"].get("enc_drop_order", "lowest")
    pred_drop_order = args["mask"].get("pred_drop_order", "lowest")
    # ng_multiblock-specific knobs (noise-weighted top-left corner sampling, Option A)
    score_mode = args["mask"].get("score_mode", "boxsum")  # "boxsum" | "corner"
    enc_bias = args["mask"].get("enc_bias", "high")  # "high" | "low" | "none"
    pred_bias = args["mask"].get("pred_bias", "high")  # "high" | "low" | "none"
    noise_temperature = float(args["mask"].get("noise_temperature", 0.5))
    # --

    # -- OPTIMIZATION
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]

    # -- LOGGING
    logging_root = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]
    tracking_info = args.setdefault("_tracking", {})
    experiment_dir = ensure_dir(os.path.join(logging_root, tag))
    runs_root = ensure_dir(os.path.join(experiment_dir, "runs"))
    run_id = (
        tracking_info.get("run_id")
        or os.environ.get("IJEPA_RUN_ID")
        or build_run_id(tag)
    )
    run_dir = ensure_dir(os.path.join(runs_root, run_id))
    params_path = os.path.join(run_dir, "params-ijepa.yaml")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    run_started_at = timestamp_utc()
    tracking_info.update(
        {
            "task_type": "pretraining",
            "experiment_tag": tag,
            "run_id": run_id,
            "output_dir": run_dir,
            "logging_root": logging_root,
            "experiment_dir": experiment_dir,
            "runs_root": runs_root,
        }
    )
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    runtime_context = get_runtime_context()
    csv_log_path = os.path.join(run_dir, f"{tag}_r{rank}.csv")
    run_latest_path = os.path.join(run_dir, f"{tag}-latest.pth.tar")
    experiment_latest_path = os.path.join(experiment_dir, f"{tag}-latest.pth.tar")
    latest_run_info_path = os.path.join(experiment_dir, "latest_run.json")
    save_path = os.path.join(run_dir, tag + "-ep{epoch}.pth.tar")

    def resolve_load_path():
        if r_file is None:
            return experiment_latest_path
        if os.path.isabs(r_file):
            return r_file

        candidates = [
            os.path.join(run_dir, r_file),
            os.path.join(experiment_dir, r_file),
            os.path.join(logging_root, r_file),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    load_path = resolve_load_path() if load_model else None

    def write_latest_run_info(status, extra=None):
        payload = {
            "task_type": "pretraining",
            "status": status,
            "experiment_tag": tag,
            "run_id": run_id,
            "experiment_dir": experiment_dir,
            "output_dir": run_dir,
            "run_manifest_path": manifest_path,
            "run_latest_checkpoint_path": run_latest_path,
            "latest_checkpoint_path": experiment_latest_path,
            "updated_at": timestamp_utc(),
        }
        payload.update(runtime_context)
        if extra:
            payload.update(extra)
        write_json(
            latest_run_info_path,
            {key: value for key, value in payload.items() if value not in (None, "")},
        )

    def write_run_manifest(status, extra=None):
        payload = {
            "task_type": "pretraining",
            "status": status,
            "run_id": run_id,
            "experiment_tag": tag,
            "run_tag": tag,
            "output_dir": run_dir,
            "logging_root": logging_root,
            "experiment_dir": experiment_dir,
            "runs_root": runs_root,
            "config_path": tracking_info.get("config_path"),
            "launcher": tracking_info.get("launcher"),
            "params_path": params_path,
            "csv_log_path_rank0": os.path.join(run_dir, f"{tag}_r0.csv"),
            "run_latest_checkpoint_path": run_latest_path,
            "latest_checkpoint_path": experiment_latest_path,
            "latest_run_info_path": latest_run_info_path,
            "checkpoint_pattern": os.path.join(run_dir, f"{tag}-ep{{epoch}}.pth.tar"),
            "load_path": load_path,
            "resume_preempt": resume_preempt,
            "model_name": model_name,
            "mask_type": mask_type,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "train_fraction": train_fraction,
            "world_size": world_size,
            "started_at": run_started_at,
        }
        payload.update(runtime_context)
        if extra:
            payload.update(extra)
        write_json(
            manifest_path,
            {key: value for key, value in payload.items() if value not in (None, "")},
        )

    if rank == 0:
        with open(params_path, "w") as f:
            yaml.dump(args, f)
        write_run_manifest("running")
        write_latest_run_info("running")

    # Log GPU memory status after distributed init
    if torch.cuda.is_available() and rank == 0:
        logger.info("=" * 60)
        logger.info("GPU Memory Status (after init):")
        logger.info(
            f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        logger.info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        free_memory = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved(0)
        ) / 1024**3
        logger.info(f"  Free: {free_memory:.2f} GB")
        logger.info("=" * 60)

        # Warn if memory is already heavily used (shouldn't happen in SLURM, but good to check)
        memory_used_pct = (
            torch.cuda.memory_reserved(0)
            / torch.cuda.get_device_properties(0).total_memory
        ) * 100
        if memory_used_pct > 10:
            logger.warning(
                f"WARNING: {memory_used_pct:.1f}% of GPU memory already in use at startup!"
            )
            logger.warning(
                "This is unusual for SLURM jobs. Check nvidia-smi output above."
            )

    # -- make csv_logger
    csv_logger = CSVLogger(
        csv_log_path,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "mask-A"),
        ("%.5f", "mask-B"),
        ("%d", "time (ms)"),
    )

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )

    # Log memory after encoder/predictor initialization
    if torch.cuda.is_available() and rank == 0:
        mem_after_encoder = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"Memory after encoder/predictor init: {mem_after_encoder:.2f} GB")

    target_encoder = copy.deepcopy(encoder)

    # Log memory after target_encoder creation
    if torch.cuda.is_available() and rank == 0:
        mem_after_target = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(
            f"Memory after target_encoder (deepcopy): {mem_after_target:.2f} GB"
        )
        logger.info(
            f"  -> Target encoder added: {mem_after_target - mem_after_encoder:.2f} GB"
        )

    # -- make data transforms
    if mask_type == "multinoise":
        if green_noise_data_path is None:
            raise ValueError("green_noise_data_path must be specified when using multinoise mask type")
        mask_collator = NoiseMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
            debug_log=os.environ.get("LOG_MULTIBLOCK_DEBUG", "") == "1",
            color_noise_path=green_noise_data_path,
            color_mask_ratio=color_mask_ratio,
            enc_drop_order=enc_drop_order,
            pred_drop_order=pred_drop_order,
        )
    elif mask_type == "quadrantnoise":
        if green_noise_data_path is None:
            raise ValueError(
                "green_noise_data_path must be specified when using quadrantnoise mask type"
            )
        mask_collator = QuadrantNoiseMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
            debug_log=os.environ.get("LOG_MULTIBLOCK_DEBUG", "") == "1",
            color_noise_path=green_noise_data_path,
            color_mask_ratio=color_mask_ratio,
            enc_drop_order=enc_drop_order,
            pred_drop_order=pred_drop_order,
        )
    elif mask_type == "ng_multiblock":
        if green_noise_data_path is None:
            raise ValueError(
                "green_noise_data_path must be specified when using ng_multiblock mask type"
            )
        mask_collator = NGMBMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
            debug_log=os.environ.get("LOG_MULTIBLOCK_DEBUG", "") == "1",
            color_noise_path=green_noise_data_path,
            score_mode=score_mode,
            enc_bias=enc_bias,
            pred_bias=pred_bias,
            noise_temperature=noise_temperature,
        )
    else:
        mask_collator = MBMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            pred_mask_scale=pred_mask_scale,
            enc_mask_scale=enc_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            allow_overlap=allow_overlap,
            min_keep=min_keep,
            debug_log=os.environ.get("LOG_MULTIBLOCK_DEBUG", "") == "1",
        )

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
    )

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=True,
        train_fraction=train_fraction,
    )
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Log memory after DDP wrapping
    if torch.cuda.is_available() and rank == 0:
        mem_after_ddp = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"Memory after DDP wrapping: {mem_after_ddp:.2f} GB")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Total GPU memory: {total_mem:.2f} GB")
        logger.info(
            f"Memory usage: {mem_after_ddp / total_mem * 100:.1f}% (before training loop)"
        )
        if mem_after_ddp / total_mem > 0.8:
            logger.error("ERROR: Model already uses >80% of GPU memory!")
            logger.error("This will likely cause OOM during training.")
            logger.error("Solutions:")
            logger.error("  1. Use a smaller model (vit_base instead of vit_huge)")
            logger.error("  2. Request a GPU with more memory")
            logger.error("  3. Use gradient checkpointing (not implemented)")
            raise RuntimeError("Model too large for available GPU memory")

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = (
            load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
            )
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, run_latest_path)
            torch.save(save_dict, experiment_latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)

            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Clear cache after backward to free memory
                torch.cuda.empty_cache()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(
                        encoder.parameters(), target_encoder.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                # Clear cache after momentum update
                torch.cuda.empty_cache()

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # Log memory usage periodically
            if (
                torch.cuda.is_available()
                and rank == 0
                and (itr % log_freq == 0 or itr == 0)
            ):
                mem_used = torch.cuda.memory_reserved(0) / 1024**3
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(
                    f"[Memory] Reserved: {mem_used:.2f} GB, "
                    f"Allocated: {mem_allocated:.2f} GB, "
                    f"Usage: {mem_used / total_mem * 100:.1f}%"
                )

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime
                )
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %.1f %.1f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "(%.1f ms)"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            maskA_meter.avg,
                            maskB_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            time_meter.avg,
                        )
                    )

                    if grad_stats is not None:
                        logger.info(
                            "[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                            )
                        )

            log_stats()

            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint after every epoch
        logger.info("avg. loss %.3f" % loss_meter.avg)
        save_checkpoint(epoch + 1)

    if rank == 0:
        write_run_manifest(
            "completed",
            {
                "completed_at": timestamp_utc(),
                "epochs_completed": num_epochs,
                "final_loss": loss_meter.avg,
            },
        )
        write_latest_run_info(
            "completed",
            {
                "completed_at": timestamp_utc(),
                "final_loss": loss_meter.avg,
            },
        )


if __name__ == "__main__":
    main()
