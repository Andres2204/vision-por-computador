"""Shared utilities for the landuse classification challenge.

Every notebook imports this module so that:

* images are loaded in the same order,
* the train/validation/test split is identical,
* metrics and artifacts follow the same schema.

Notebooks run with ``guia-4/reto-2`` as the working directory. Artifacts are
stored under ``resultados/`` (the Spanish folder name is kept for continuity),
while all Python identifiers, comments and file formats inside this module are
in English.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "landuse-dataset"
RESULTS_DIR = BASE_DIR / "resultados"
SPLIT_PATH = RESULTS_DIR / "split.json"

CLASSES = ["denseresidential", "harbor", "river"]
IMG_SIZE = 128
SEED = 20
TEST_SIZE = 0.25
VAL_SIZE = 0.15  # fraction of the training set reserved for validation

# Artifact names used by every model folder.
ARTIFACTS = {
    "metrics": "metrics.json",
    "predictions": "predictions.csv",
    "history": "history.csv",
    "report": "classification_report.txt",
    "model": "model.keras",
    "weights": "model.weights.h5",
    "torch_model": "model.pt",
}


# ---------------------------------------------------------------------------
# Data loading and split
# ---------------------------------------------------------------------------
def load_images(size: int = IMG_SIZE, classes: list[str] | None = None):
    """Load the local dataset in alphabetical order.

    Returns ``(images, labels, paths)``. Images are RGB ``uint8`` arrays of
    shape ``(size, size, 3)`` and are not normalized, so each model can apply
    its own preprocessing.
    """
    classes = classes or CLASSES
    images, labels, paths = [], [], []
    for class_name in classes:
        folder = DATASET_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for path in sorted(folder.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[warning] could not read {path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(class_name)
            paths.append(str(path.relative_to(BASE_DIR)))
    return np.asarray(images, dtype=np.uint8), np.asarray(labels), paths


def get_split(y, paths, seed: int = SEED):
    """Return the shared train/validation/test indices.

    The split is written to ``resultados/split.json`` the first time and
    reused afterwards. It is stratified and reproducible.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SPLIT_PATH.exists():
        data = json.loads(SPLIT_PATH.read_text())
        return (
            np.array(data["train_idx"]),
            np.array(data["val_idx"]),
            np.array(data["test_idx"]),
        )

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE,
        stratify=y[train_val_idx],
        random_state=seed,
    )

    data = {
        "seed": seed,
        "test_size": TEST_SIZE,
        "val_size_of_train": VAL_SIZE,
        "classes": CLASSES,
        "n_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "test_paths": [paths[i] for i in test_idx],
        "test_class_counts": {
            class_name: int(np.sum(y[test_idx] == class_name))
            for class_name in CLASSES
        },
    }
    SPLIT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return train_idx, val_idx, test_idx


def load_split_data(size: int = IMG_SIZE):
    """Load images and apply the shared split."""
    X, y, paths = load_images(size)
    train_idx, val_idx, test_idx = get_split(y, paths)
    return {
        "X": X,
        "y": y,
        "paths": paths,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "test_paths": [paths[i] for i in test_idx],
    }


def one_hot(labels, classes: list[str] | None = None):
    """Convert string labels to a one-hot matrix following ``classes`` order."""
    import keras

    classes = classes or CLASSES
    return keras.utils.to_categorical(
        [classes.index(label) for label in labels], len(classes)
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_keras_model(model, X_train, y_train, X_val, y_val, **fit_kwargs):
    """Fit a Keras model and return ``(history, seconds)``."""
    started = time.perf_counter()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), verbose=2, **fit_kwargs
    )
    seconds = time.perf_counter() - started
    return history, seconds


def predict_keras(model, X_test, batch_size: int = 32):
    """Return predicted indices, probabilities and milliseconds per image."""
    started = time.perf_counter()
    probabilities = model.predict(X_test, batch_size=batch_size, verbose=0)
    seconds = time.perf_counter() - started
    return (
        probabilities.argmax(axis=1),
        probabilities,
        1000 * seconds / len(X_test),
    )


def compute_metrics(
    name: str,
    family: str,
    y_true,
    y_pred,
    y_prob=None,
    classes: list[str] | None = None,
    params=None,
    trainable_params=None,
    training_seconds=None,
    inference_ms=None,
    epochs=None,
    hyperparameters: dict | None = None,
    notes: str = "",
):
    """Build the metrics dictionary stored in ``metrics.json``."""
    classes = classes or CLASSES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "name": name,
        "family": family,
        "classes": classes,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": {},
        "parameters": None if params is None else int(params),
        "trainable_parameters": (
            None if trainable_params is None else int(trainable_params)
        ),
        "training_seconds": (
            None if training_seconds is None else round(float(training_seconds), 2)
        ),
        "inference_ms_per_image": (
            None if inference_ms is None else round(float(inference_ms), 3)
        ),
        "epochs": None if epochs is None else int(epochs),
        "hyperparameters": hyperparameters or {},
        "notes": notes,
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    for class_name in classes:
        metrics["per_class"][class_name] = {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"]),
        }

    if y_prob is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    np.asarray(y_prob),
                    multi_class="ovr",
                    average="macro",
                    labels=classes,
                )
            )
        except ValueError as exc:  # a class may be missing from the test set
            metrics["roc_auc_ovr_macro"] = None
            metrics["notes"] += f" ROC-AUC not computable: {exc}"
    else:
        metrics["roc_auc_ovr_macro"] = None

    return metrics


def save_results(
    folder: str | Path,
    metrics: dict,
    y_true,
    y_pred,
    y_prob=None,
    test_paths=None,
    history=None,
    model=None,
):
    """Save metrics, predictions, report, history and model for one model.

    ``folder`` is relative to ``resultados/`` (for example ``cnn/base``).
    Per-model figures are intentionally not generated: the comparison notebook
    builds the four figures used in the report from these artifacts.
    """
    destination = RESULTS_DIR / folder
    destination.mkdir(parents=True, exist_ok=True)

    (destination / ARTIFACTS["metrics"]).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )

    with open(destination / ARTIFACTS["report"], "w", encoding="utf-8") as handle:
        handle.write(
            classification_report(
                y_true,
                y_pred,
                labels=metrics["classes"],
                target_names=metrics["classes"],
                digits=4,
                zero_division=0,
            )
        )

    # Predictions with probabilities: everything can be recomputed without
    # retraining.
    frame = pd.DataFrame(
        {
            "path": test_paths if test_paths is not None else np.arange(len(y_true)),
            "true_class": y_true,
            "predicted_class": y_pred,
        }
    )
    if y_prob is not None:
        for index, class_name in enumerate(metrics["classes"]):
            frame[f"prob_{class_name}"] = np.asarray(y_prob)[:, index]
    frame.to_csv(destination / ARTIFACTS["predictions"], index=False)

    if history is not None:
        if isinstance(history, pd.DataFrame):
            history_frame = history.copy()
        else:
            history_frame = pd.DataFrame(history.history)
        history_frame.index = np.arange(1, len(history_frame) + 1)
        history_frame.index.name = "epoch"
        history_frame.to_csv(destination / ARTIFACTS["history"])

    if model is not None:
        try:
            model.save(destination / ARTIFACTS["model"])
        except Exception as exc:  # custom layers without get_config()
            print(f"[warning] full model could not be saved: {exc}")
            model.save_weights(destination / ARTIFACTS["weights"])

    print(f"Results saved to: {destination}")
    return destination


# ---------------------------------------------------------------------------
# Artifact loading and load-or-retrain helper
# ---------------------------------------------------------------------------
def result_dir(folder: str | Path) -> Path:
    """Return the absolute path of a model folder inside ``resultados/``."""
    return RESULTS_DIR / folder


def load_split():
    """Load ``split.json`` as a dictionary, or an empty dict if absent."""
    if not SPLIT_PATH.exists():
        return {}
    return json.loads(SPLIT_PATH.read_text())


def load_metrics(folder: str | Path):
    """Load ``metrics.json`` for a model folder, or ``None`` if absent."""
    path = result_dir(folder) / ARTIFACTS["metrics"]
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_history(folder: str | Path):
    """Load ``history.csv`` as a DataFrame indexed by epoch, or ``None``."""
    path = result_dir(folder) / ARTIFACTS["history"]
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="epoch")


def load_predictions(folder: str | Path):
    """Load ``predictions.csv`` for a model folder, or ``None`` if absent."""
    path = result_dir(folder) / ARTIFACTS["predictions"]
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_saved_model(folder: str | Path):
    """Load a full Keras model. Custom architectures need their own loader."""
    import keras

    path = result_dir(folder) / ARTIFACTS["model"]
    if not path.exists():
        return None
    return keras.models.load_model(path)


def load_or_train(
    folder: str | Path,
    build_model,
    fit_model,
    evaluate_model,
    load_model=None,
    force: bool = False,
):
    """Load a cached model or train it from scratch.

    Parameters
    ----------
    folder:
        Model folder relative to ``resultados/``.
    build_model:
        Zero-argument callable returning a compiled or uncompiled model.
    fit_model:
        Callable ``fit_model(model) -> (model, history, training_seconds)``.
    evaluate_model:
        Callable ``evaluate_model(model, history, training_seconds) -> metrics``.
        It is only called after training and is responsible for saving results.
    load_model:
        Optional callable ``load_model(folder) -> model`` for architectures
        that need a custom loader instead of ``.keras`` deserialization.
    force:
        When ``True`` the model is retrained even if a cached copy exists.

    Returns
    -------
    ``(model, history, metrics, retrained)``.
    """
    folder_path = result_dir(folder)
    metrics = load_metrics(folder)
    history = load_history(folder)
    has_artifact = (
        (folder_path / ARTIFACTS["model"]).exists()
        or (folder_path / ARTIFACTS["weights"]).exists()
    )

    if not force and metrics is not None and has_artifact:
        try:
            model = load_model(folder) if load_model is not None else load_saved_model(folder)
        except Exception as exc:
            print(f"[cache] Could not load model ({exc}); retraining.")
            model = None
        if model is not None:
            print(f"[cache] Using trained model from {folder_path}")
            return model, history, metrics, False

    print(f"[train] Training model in {folder_path}")
    model = build_model()
    model, history, training_seconds = fit_model(model)
    metrics = evaluate_model(model, history, training_seconds)
    return model, history, metrics, True


# ---------------------------------------------------------------------------
# Aggregated results and statistical helpers
# ---------------------------------------------------------------------------
def load_all_results():
    """Return one row per ``metrics.json`` found under ``resultados/``."""
    rows = []
    for path in sorted(RESULTS_DIR.rglob(ARTIFACTS["metrics"])):
        data = json.loads(path.read_text())
        folder = path.parent.relative_to(RESULTS_DIR)
        parts = folder.parts
        row = {
            "group": parts[0],
            "variant": parts[1] if len(parts) > 1 else "",
            "folder": str(folder),
            "model": data["name"],
            "family": data["family"],
            "accuracy": data["accuracy"],
            "f1_macro": data["f1_macro"],
            "f1_weighted": data["f1_weighted"],
            "precision_macro": data["precision_macro"],
            "recall_macro": data["recall_macro"],
            "cohen_kappa": data["cohen_kappa"],
            "roc_auc_ovr_macro": data["roc_auc_ovr_macro"],
            "parameters": data["parameters"],
            "training_seconds": data["training_seconds"],
            "inference_ms_per_image": data["inference_ms_per_image"],
            "epochs": data["epochs"],
        }
        for class_name, values in data["per_class"].items():
            row[f"f1_{class_name}"] = values["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_accuracy_ci(y_true, y_pred, n_boot: int = 2000, seed: int = SEED):
    """Return the 2.5 and 97.5 percentiles of the bootstrap accuracy."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracies = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y_true), len(y_true))
        accuracies.append(np.mean(y_true[indices] == y_pred[indices]))
    return tuple(np.percentile(accuracies, [2.5, 97.5]))


def mcnemar_pvalue(y_true, pred_a, pred_b):
    """Exact McNemar p-value for two paired prediction vectors."""
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    total = only_a + only_b
    if total == 0:
        return 1.0
    return float(binomtest(min(only_a, only_b), total, 0.5).pvalue)
