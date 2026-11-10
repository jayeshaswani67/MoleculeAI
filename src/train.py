import sys
import random
import joblib

from pathlib import Path
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_dataset
from src.molecule_features import batch_features
from src.evaluate import evaluate_model, save_metrics


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def get_scaffold(smiles):

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=True
    )

    # Acyclic molecules need their own groups.
    if not scaffold:
        return f"ACYCLIC:{smiles}"

    return scaffold


def scaffold_split(df, seed=42):

    groups = defaultdict(list)

    for index, smiles in enumerate(df["mol"]):
        scaffold = get_scaffold(smiles)

        if scaffold is not None:
            groups[scaffold].append(index)

    scaffold_groups = list(groups.values())
    random.Random(seed).shuffle(scaffold_groups)

    n = len(df)
    train_target = int(n * 0.8)
    valid_target = int(n * 0.1)

    train_indices = []
    valid_indices = []
    test_indices = []

    for group in scaffold_groups:

        if len(train_indices) < train_target:
            train_indices.extend(group)

        elif len(valid_indices) < valid_target:
            valid_indices.extend(group)

        else:
            test_indices.extend(group)

    return train_indices, valid_indices, test_indices


def train():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()

    print("\nGenerating molecular features...")
    X = batch_features(df["mol"].tolist())
    y = df["Class"].values

    train_idx, valid_idx, test_idx = scaffold_split(df)

    if not train_idx or not valid_idx or not test_idx:
        raise ValueError("Scaffold split produced an empty dataset partition.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\nDataset split:")
    print("Training:", len(train_idx))
    print("Validation:", len(valid_idx))
    print("Testing:", len(test_idx))

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nValidation performance:")
    valid_metrics = evaluate_model(model, X_valid, y_valid)

    print("\nFinal test performance:")
    test_metrics = evaluate_model(model, X_test, y_test)

    model_path = MODEL_DIR / "moleculeai_bace.joblib"

    joblib.dump(model, model_path)

    save_metrics(
        test_metrics,
        MODEL_DIR / "evaluation_metrics.json"
    )

    print("\nTraining complete!")
    print("Model saved:", model_path)

    return model


if __name__ == "__main__":
    train()
