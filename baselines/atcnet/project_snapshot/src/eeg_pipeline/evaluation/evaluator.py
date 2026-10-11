"""Test-set evaluation for EEG classification models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
) -> dict[str, float | list[list[int]]]:
    """Evaluate a model on the test set in one pass."""
    model.eval()
    total_loss = 0.0
    total_trials = 0
    n_classes = 0
    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    with torch.inference_mode():
        for eeg, labels in test_loader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(eeg)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_trials += batch_size
            n_classes = logits.size(1)
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    if total_trials == 0:
        raise ValueError("test_loader must contain at least one trial")

    true_labels = torch.cat(all_labels).numpy()
    predicted_labels = torch.cat(all_predictions).numpy()
    classes = list(range(n_classes))

    return {
        "loss": total_loss / total_trials,
        "accuracy": float((true_labels == predicted_labels).mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=classes,
        ).tolist(),
    }


def save_evaluation_result(
    result: dict[str, float | list[list[int]]],
    output_dir: str | Path,
) -> Path:
    """Save test metrics as a readable JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "test_metrics.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result_path


if __name__ == "__main__":
    import math

    from torch.utils.data import TensorDataset

    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=3)
    result = evaluate_model(
        nn.Identity(), loader, nn.CrossEntropyLoss(), device="cpu"
    )

    output_dir = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "evaluator_smoke_test"
        / "identity"
        / "fold_01"
    )
    result_path = save_evaluation_result(result, output_dir)
    with result_path.open(encoding="utf-8") as file:
        saved_result = json.load(file)
    if saved_result != result:
        raise RuntimeError("Saved evaluation result does not match input")

    expected_loss = nn.CrossEntropyLoss()(logits, labels).item()
    if not math.isclose(result["loss"], expected_loss, rel_tol=1e-6):
        raise RuntimeError(f"Incorrect loss: {result['loss']}")
    if result["accuracy"] != 0.75:
        raise RuntimeError(f"Incorrect accuracy: {result['accuracy']}")
    if result["confusion_matrix"] != [[1, 1, 0], [0, 1, 0], [0, 0, 1]]:
        raise RuntimeError(
            f"Incorrect confusion matrix: {result['confusion_matrix']}"
        )

    print("evaluate_model test passed")
    print(result)
    print(f"saved to: {result_path}")
