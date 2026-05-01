import csv
import os
import logging

from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.utils.plot import plot_loss_curves


logger = logging.getLogger(__name__)


def _reduce_metrics(correct, total, loss_sum, device):
    """All-reduce [correct, total, loss_sum] across ranks.

    No-op when a distributed process group is not active.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return correct, total, loss_sum
    t = torch.tensor([correct, total, loss_sum],
                     dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t[0].item()), int(t[1].item()), t[2].item()


def train_linear_probe(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    device,
    outputs_dir,
    rank=0,
    world_size=1,
    train_sampler=None,
):
    """Training loop with optional distributed support.

    When *rank*, *world_size*, and *train_sampler* are left at their
    defaults the function behaves identically to the original single-GPU
    version (full backward compatibility with ``main_linprobe.py``).
    """
    is_distributed = world_size > 1
    is_main = rank == 0

    logger.info("Starting linear probe training  "
                "(rank=%d, world_size=%d, distributed=%s)",
                rank, world_size, is_distributed)

    metrics_path = os.path.join(outputs_dir, "metrics.csv")
    best_checkpoint_path = os.path.join(outputs_dir, "best_linear_probe.pth")
    plot_path = os.path.join(outputs_dir, "training_loss_plot.png")

    if is_main:
        with open(metrics_path, "w", newline="") as metrics_handle:
            writer = csv.writer(metrics_handle)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_acc",
                    "val_acc",
                    "lr",
                    "best_val_acc_so_far",
                ]
            )

    inner = model.module if hasattr(model, "module") else model
    inner.encoder.eval()
    model = model.to(device)

    optimizer = torch.optim.SGD(
        inner.classifier.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        f"Hyperparameters: lr={learning_rate}, epochs={num_epochs}, "
        f"optimizer={optimizer.__class__.__name__}(weight_decay={optimizer.defaults['weight_decay']}), "
        f"scheduler={scheduler.__class__.__name__}(step_size={scheduler.step_size}, gamma={scheduler.gamma}), "
        f"criterion={criterion.__class__.__name__}, device={device}"
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Train loader has 0 batches. "
            "Check that train_frac * num_samples >= batch_size."
        )

    # --- TRAINING ---
    best_acc = 0.0
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        inner.classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]["lr"]

        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                          disable=(not is_main))
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if epoch == 0 and batch_idx == 0:
                _, preds = outputs.max(1)
                logger.info(f"[DIAG] Train batch 0 — labels[:20]: {labels[:20].tolist()}")
                logger.info(f"[DIAG] Train batch 0 — preds[:20]:  {preds[:20].tolist()}")
                logger.info(f"[DIAG] Train batch 0 — unique labels: {labels.unique().numel()}, unique preds: {preds.unique().numel()}")
                logger.info(f"[DIAG] Train batch 0 — loss: {loss.item():.6f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Same as val: skip tqdm postfix under DDP (counts are rank-local until all_reduce).
            if is_main and not is_distributed:
                avg_loss = train_loss / train_total
                avg_acc = 100.0 * train_correct / train_total
                train_pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    acc=f"{avg_acc:.2f}%",
                )

        # Reduce train metrics across ranks
        train_correct, train_total, train_loss = _reduce_metrics(
            train_correct, train_total, train_loss, device)
        num_train_batches = len(train_loader) * world_size
        train_losses.append(train_loss / num_train_batches)
        scheduler.step()

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader,
                            desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                            disable=(not is_main))
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                if epoch == 0 and batch_idx == 0:
                    _, preds = outputs.max(1)
                    logger.info(f"[DIAG] Val batch 0 — labels[:20]:  {labels[:20].tolist()}")
                    logger.info(f"[DIAG] Val batch 0 — preds[:20]:   {preds[:20].tolist()}")
                    logger.info(f"[DIAG] Val batch 0 — unique labels: {labels.unique().numel()}, unique preds: {preds.unique().numel()}")
                    logger.info(f"[DIAG] Val batch 0 — loss: {loss.item():.6f}")

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                if is_main and not is_distributed:
                    val_pbar.set_postfix(acc=f"{100.0 * val_correct / val_total:.2f}%")

        # Reduce val metrics across ranks
        val_correct, val_total, val_loss = _reduce_metrics(
            val_correct, val_total, val_loss, device)
        num_val_batches = len(val_loader) * world_size
        val_losses.append(val_loss / num_val_batches)

        # Compute global accuracies (identical on all ranks after reduce)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model (rank 0 only)
        if val_acc > best_acc:
            logger.info(f"New best model! ({val_acc:.2f}% > {best_acc:.2f}%)")
            best_acc = val_acc
            if is_main:
                torch.save(
                    {
                        "classifier": inner.classifier.state_dict(),
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                    },
                    best_checkpoint_path,
                )

        if is_main:
            with open(metrics_path, "a", newline="") as metrics_handle:
                writer = csv.writer(metrics_handle)
                writer.writerow(
                    [
                        epoch + 1,
                        f"{train_losses[-1]:.6f}",
                        f"{val_losses[-1]:.6f}",
                        f"{train_acc:.4f}",
                        f"{val_acc:.4f}",
                        f"{current_lr:.8f}",
                        f"{best_acc:.4f}",
                    ]
                )

    logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

    if is_main:
        plot_loss_curves(
            train_losses,
            val_losses,
            num_epochs,
            save_path=plot_path,
        )

    return {
        "best_acc": best_acc,
        "best_checkpoint_path": best_checkpoint_path,
        "metrics_path": metrics_path,
        "plot_path": plot_path,
    }
