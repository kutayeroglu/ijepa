import os
import logging

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.utils.plot import plot_loss_curves


logger = logging.getLogger(__name__)


def train_linear_probe(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    device,
    outputs_dir,
):
    """Training loop"""
    logger.info("Starting linear probe training")

    # Mode configuration
    model.encoder.eval()
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
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
        model.classifier.train()  # Train only the classifier head
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
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

            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_pbar.set_postfix(
                loss=f"{train_loss / train_total:.4f}",
                acc=f"{100.0 * train_correct / train_total:.2f}%",
            )

        train_losses.append(train_loss / len(train_loader))
        scheduler.step()

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
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
                val_pbar.set_postfix(acc=f"{100.0 * val_correct / val_total:.2f}%")

        val_losses.append(val_loss / len(val_loader))

        # Log epoch results
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model
        if val_acc > best_acc:
            logger.info(f"New best model! ({val_acc:.2f}% > {best_acc:.2f}%)")
            best_acc = val_acc
            checkpoint_path = os.path.join(outputs_dir, "best_linear_probe.pth")
            torch.save(
                {
                    "classifier": model.classifier.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

    logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

    plot_path = os.path.join(outputs_dir, "training_loss_plot.png")
    plot_loss_curves(
        train_losses,
        val_losses,
        num_epochs,
        save_path=plot_path,
    )

    return best_acc
