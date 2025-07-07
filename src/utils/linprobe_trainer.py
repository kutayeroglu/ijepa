import torch
import torch.nn as nn


def train_linear_probe(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
):
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.parameters(), lr=learning_rate, momentum=0.9
    )

    model = model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss / len(train_loader):.4f} - Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_linear_probe.pth")
