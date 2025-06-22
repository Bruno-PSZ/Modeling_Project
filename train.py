import torch
import torch.nn as nn
from tqdm import tqdm
from utils import compute_metrics


def train_model(
    model, model_name, train_loader, val_loader, optimizer, num_epochs, device, task
):
    """
    Trains the model and optionally logs metrics to ClearML.

    Parameters:
        model (nn.Module): PyTorch model
        model_name (str): Name of the model for logging
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        device (torch.device): CUDA or CPU
        task (clearml.Task or None): Active ClearML task or None
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # ---- Initial validation before training ----
    model.eval()
    initial_preds, initial_targets = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Initial Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            initial_preds.extend(preds.cpu().numpy())
            initial_targets.extend(targets.cpu().numpy())

    compute_metrics(
        initial_preds,
        initial_targets,
        prefix="Val",
        epoch=0,
        task=task,
        model_name=model_name,
    )

    # ---- Training loop ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        pbar = tqdm(
            train_loader, desc=f"{model_name} | Epoch {epoch}/{num_epochs} [Train]"
        )
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        if task is not None:
            task.get_logger().report_scalar(
                "Loss_Train", model_name, iteration=epoch, value=avg_train_loss
            )
        else:
            print(f"[Train][Epoch {epoch}] Loss: {avg_train_loss:.4f}")

        train_metrics = compute_metrics(
            train_preds,
            train_targets,
            prefix="Train",
            epoch=epoch,
            task=task,
            model_name=model_name,
        )

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)

        if task is not None:
            task.get_logger().report_scalar(
                "Loss_Val", model_name, iteration=epoch, value=avg_val_loss
            )
        else:
            print(f"[Val][Epoch {epoch}] Loss: {avg_val_loss:.4f}")

        val_metrics = compute_metrics(
            val_preds,
            val_targets,
            prefix="Val",
            epoch=epoch,
            task=task,
            model_name=model_name,
        )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Acc: {train_metrics['acc']:.4f} | Val Acc: {val_metrics['acc']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )
