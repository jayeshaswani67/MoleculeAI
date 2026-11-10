from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

from src.molecule_features import smiles_to_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "bace.csv"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42


def get_scaffold(smiles):
    """Return a Bemis-Murcko scaffold for a valid molecule."""
    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=True,
    )

    # Group acyclic molecules together by canonical SMILES.
    if not scaffold:
        return Chem.MolToSmiles(molecule, canonical=True)

    return scaffold


def scaffold_split(smiles, train_ratio=0.8, validation_ratio=0.1):
    """
    Split data by scaffold groups.
    Molecules sharing a scaffold remain in the same partition.
    """
    groups = {}

    for index, smiles_value in enumerate(smiles):
        scaffold = get_scaffold(smiles_value)

        if scaffold is not None:
            groups.setdefault(scaffold, []).append(index)

    rng = np.random.default_rng(RANDOM_STATE)
    scaffold_groups = list(groups.values())
    rng.shuffle(scaffold_groups)

    total = len(smiles)
    train_indices = []
    validation_indices = []
    test_indices = []

    for group in scaffold_groups:
        if len(train_indices) < total * train_ratio:
            train_indices.extend(group)
        elif len(validation_indices) < total * validation_ratio:
            validation_indices.extend(group)
        else:
            test_indices.extend(group)

    return (
        np.array(train_indices, dtype=int),
        np.array(validation_indices, dtype=int),
        np.array(test_indices, dtype=int),
    )


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python -m src.data_loader"
        )

    df = pd.read_csv(DATA_PATH)

    if "mol" not in df.columns or "Class" not in df.columns:
        raise ValueError(
            "Expected columns 'mol' and 'Class' in bace.csv."
        )

    df = df[["mol", "Class"]].dropna().copy()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)

    df = df[df["Class"].isin([0, 1])]
    df = df.drop_duplicates(subset=["mol"]).reset_index(drop=True)

    valid_rows = []

    for index, smiles in enumerate(df["mol"]):
        if Chem.MolFromSmiles(str(smiles)) is not None:
            valid_rows.append(index)

    df = df.iloc[valid_rows].reset_index(drop=True)

    if df["Class"].nunique() != 2:
        raise ValueError("Dataset must contain both activity classes.")

    return df


def create_feature_matrix(smiles_values):
    """Use the existing MoleculeAI feature extractor."""
    features = []

    for smiles in smiles_values:
        vector = smiles_to_features(smiles)

        if vector is None:
            raise ValueError(f"Feature extraction failed for: {smiles}")

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        features.append(vector)

    return np.vstack(features)


def calculate_metrics(y_true, predictions, probabilities):
    """Calculate classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y_true, predictions)
        ),
        "roc_auc": None,
        "pr_auc": None,
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
        metrics["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )

    return metrics


def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def run_benchmark():
    warnings.filterwarnings("ignore", category=FutureWarning)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nMOLECULEAI MODEL BENCHMARK")
    print("=" * 55)

    df = load_dataset()

    print(f"Dataset size: {len(df)}")
    print("\nExtracting molecular features...")

    X = create_feature_matrix(df["mol"].tolist())
    y = df["Class"].to_numpy()

    print(f"Feature matrix shape: {X.shape}")

    train_idx, validation_idx, test_idx = scaffold_split(
        df["mol"].tolist()
    )

    if not len(train_idx) or not len(validation_idx) or not len(test_idx):
        raise ValueError(
            "Scaffold split produced an empty partition. "
            "Inspect the dataset and scaffold grouping."
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[validation_idx], y[validation_idx]

    print(f"\nTraining molecules: {len(train_idx)}")
    print(f"Validation molecules: {len(validation_idx)}")
    print(f"Reserved test molecules: {len(test_idx)}")

    results = []

    for name, model in get_models().items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        probabilities = model.predict_proba(X_val)[:, 1]

        metrics = calculate_metrics(
            y_val,
            predictions,
            probabilities,
        )

        row = {"model": name, **metrics}
        results.append(row)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
        na_position="last",
    )

    csv_path = REPORT_DIR / "benchmark_results.csv"
    json_path = REPORT_DIR / "benchmark_summary.json"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 55)
    print("BENCHMARK RESULTS")
    print("=" * 55)

    print(
        results_df[
            ["model", "accuracy", "precision", "recall",
             "f1", "mcc", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )

    print(f"\nCSV report: {csv_path}")
    print(f"JSON report: {json_path}")

    print(
        "\nNote: These are validation results. "
        "The reserved test partition has not been evaluated."
    )


if __name__ == "__main__":
    run_benchmark()