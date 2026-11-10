
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.data_loader import load_dataset
from src.train import scaffold_split
from src.molecule_features import batch_features, smiles_to_features


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "moleculeai_bace.joblib"
REPORT_DIR = ROOT / "data" / "processed"


def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first.")

    return joblib.load(MODEL_PATH)


def test_unseen_molecules():

    print("\nMOLECULEAI MODEL VALIDATION")
    print("=" * 45)

    model = load_trained_model()
    df = load_dataset()

    train_idx, valid_idx, test_idx = scaffold_split(df)

    test_df = df.iloc[test_idx].copy()

    X_test = batch_features(test_df["mol"].tolist())
    y_true = test_df["Class"].values

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(set(y_true)) == 2 else None
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist()
    }

    print("\nEvaluation metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    test_df["actual_class"] = y_true
    test_df["predicted_class"] = predictions
    test_df["active_probability"] = probabilities

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_df.to_csv(
        REPORT_DIR / "model_test_results.csv",
        index=False
    )

    with open(REPORT_DIR / "test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nTesting completed.")
    print("Results saved in data/processed/")


def test_single_molecule(smiles):

    model = load_trained_model()

    features = smiles_to_features(smiles).reshape(1, -1)

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)

    active_probability = float(
        probabilities[classes.index(1)]
    )

    print("\nMOLECULE PREDICTION")
    print("-" * 35)
    print("SMILES:", smiles)
    print("Predicted class:", prediction)
    print("Active-class model score:", round(active_probability, 4))

    print(
        "Interpretation:",
        "Predicted active" if prediction == 1
        else "Predicted inactive"
    )

    print("\nThis prediction is experimental-activity research only.")
    print("It does not establish drug efficacy or safety.")


if __name__ == "__main__":

    test_unseen_molecules()

    examples = [
        "CCO",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CCN(CC)CC"
    ]

    print("\nTesting individual molecular structures:")

    for smiles in examples:
        try:
            test_single_molecule(smiles)
        except Exception as error:
            print(f"Could not test {smiles}: {error}")
