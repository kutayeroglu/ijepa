import logging
import time

from tqdm import tqdm
import torch
import torch.nn as nn


# --- Setup a logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


def train_linear_probe(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
):
    """Trains a linear probe with improved logging."""
    # --- 1. Log Hyperparameters ---
    logging.info("Starting new training run")
    logging.info(
        f"Hyperparameters: num_epochs={num_epochs}, learning_rate={learning_rate}, device='{device}'"
    )

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.BatchNorm(), lr=learning_rate, momentum=0.9
    )

    model = model.to(device)
    best_acc = 0.0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # --- 2. Training Loop ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Use tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Update progress bar description
            train_pbar.set_postfix(
                loss=f"{train_loss / train_total:.4f}",
                acc=f"{100.0 * train_correct / train_total:.2f}%",
            )

        # --- 3. Validation Loop ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_pbar.set_postfix(
                    loss=f"{val_loss / val_total:.4f}",
                    acc=f"{100.0 * val_correct / val_total:.2f}%",
                )

        # --- 4. Structured Logging for the Epoch ---
        epoch_duration = time.time() - epoch_start_time
        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total

        log_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_accuracy,
            "duration_seconds": round(epoch_duration, 2),
        }
        logging.info(f"Epoch Summary: {log_metrics}")

        # --- 5. Save Best Model and Log It ---
        if val_accuracy > best_acc:
            logging.info(
                f"New best model found! Validation accuracy improved from {best_acc:.2f}% to {val_accuracy:.2f}%. Saving model..."
            )
            best_acc = val_accuracy
            torch.save(model.state_dict(), "best_linear_probe.pth")

    total_duration = time.time() - total_start_time
    logging.info(f"Training finished in {total_duration:.2f} seconds.")
    logging.info(f"Best validation accuracy: {best_acc:.2f}%")
