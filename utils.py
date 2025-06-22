from sklearn.metrics import accuracy_score, precision_score, f1_score
import torch


def compute_metrics(preds, targets, prefix, epoch, task, model_name):
    """
    Computes and optionally logs classification metrics to ClearML.

    Parameters:
        preds (list or np.array): Predicted class labels
        targets (list or np.array): True class labels
        prefix (str): Either "Train" or "Val"
        epoch (int): Epoch number for logging
        task (clearml.Task or None): ClearML Task object or None
        model_name (str): Name of the model being evaluated

    Returns:
        dict: Dictionary with accuracy, precision, and F1 scores.
    """
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    if task is not None:
        logger = task.get_logger()
        logger.report_scalar(
            f"Accuracy_{prefix}", model_name, iteration=epoch, value=acc
        )
        logger.report_scalar(
            f"Precision_{prefix}", model_name, iteration=epoch, value=prec
        )
        logger.report_scalar(f"F1_{prefix}", model_name, iteration=epoch, value=f1)

    return {"acc": acc, "prec": prec, "f1": f1}


def predict_sample(model, sample, device, label_encoder):

    model.eval()

    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)

    sample = sample.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return pred_idx, pred_label
