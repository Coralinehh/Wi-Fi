from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device, num_classes=22, print_per_class=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for index, label in enumerate(labels):
                label_id = int(label.item())
                class_total[label_id] += 1
                class_correct[label_id] += int(predicted[index] == label)

    if print_per_class:
        print("\nPer-class accuracy:")
        for index in range(num_classes):
            if class_total[index] == 0:
                print(f"  Class {index}: no samples")
            else:
                acc = 100.0 * class_correct[index] / class_total[index]
                print(f"  Class {index}: {acc:.2f}% ({class_correct[index]}/{class_total[index]})")

    return total_loss / len(loader), 100.0 * correct / total


def fit(
    model,
    loaders,
    device,
    epochs=50,
    learning_rate=0.001,
    num_classes=22,
    checkpoint_path="checkpoints/model_best.pt",
    early_stop_patience=5,
    optimizer_name="rmsprop",
    momentum=0.9,
    scheduler_name="step",
    amp_enabled=False,
):
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    scaler = amp.GradScaler() if amp_enabled else None
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_test_acc = 0.0
    best_test_loss = None
    best_val_loss = float("inf")
    no_improve_counter = 0
    start = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device, num_classes)
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device, num_classes, print_per_class=True)
        if scheduler is not None:
            scheduler.step()

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
        print(f"Val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%")
        print(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= early_stop_patience:
                print(f"Early stopping after {early_stop_patience} epochs without validation loss improvement.")
                break

    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed:.2f}s")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    if best_test_loss is not None:
        print(f"Best test loss: {best_test_loss:.4f}")
    return {
        "best_test_acc": best_test_acc,
        "best_test_loss": best_test_loss,
        "best_val_loss": best_val_loss,
        "training_time": elapsed,
    }
