from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.molecule_features import smiles_to_features


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "moleculeai_bace.joblib"
DATA_PATH = ROOT / "data" / "raw" / "bace.csv"
REPORT_DIR = ROOT / "reports"

FINGERPRINT_SIZE = 2048

FINGERPRINT_GENERATOR = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=FINGERPRINT_SIZE,
    )
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train your model first."
        )

    return joblib.load(MODEL_PATH)


def prepare_features(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    features = smiles_to_features(smiles)

    if features is None:
        raise ValueError("Could not extract molecular features.")

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def get_active_probability(model, features):
    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    active_index = classes.index(1)

    return float(probabilities[active_index])


def calculate_entropy(probabilities):
    """Normalized entropy: 0 means certain, 1 means maximum uncertainty."""
    probabilities = np.asarray(probabilities, dtype=float)

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    maximum_entropy = np.log(len(probabilities))

    if maximum_entropy == 0:
        return 0.0

    return float(entropy / maximum_entropy)


def get_tree_disagreement(model, features):
    """
    Estimate disagreement across individual Random Forest trees.

    Returns None for models without individual estimators.
    """
    estimators = getattr(model, "estimators_", None)

    if estimators is None:
        return None

    estimators = np.asarray(estimators, dtype=object).reshape(-1)

    if len(estimators) == 0:
        return None

    tree_scores = []

    for tree in estimators:
        if not hasattr(tree, "predict_proba"):
            continue

        tree_probabilities = tree.predict_proba(features)[0]
        tree_classes = list(tree.classes_)

        if 1 in tree_classes:
            score = tree_probabilities[tree_classes.index(1)]
        else:
            score = 0.0

        tree_scores.append(float(score))

    if not tree_scores:
        return None

    return {
        "mean_active_score": float(np.mean(tree_scores)),
        "standard_deviation": float(np.std(tree_scores)),
        "minimum": float(np.min(tree_scores)),
        "maximum": float(np.max(tree_scores)),
        "number_of_trees": len(tree_scores),
    }


def calculate_nearest_similarity(smiles):
    """
    Compare the query molecule against the BACE dataset.

    Returns the maximum Morgan fingerprint Tanimoto similarity.
    """
    if not DATA_PATH.exists():
        return {
            "available": False,
            "reason": "BACE dataset not found.",
        }

    query_molecule = Chem.MolFromSmiles(smiles)

    if query_molecule is None:
        raise ValueError("Invalid query molecule.")

    query_fp = FINGERPRINT_GENERATOR.GetFingerprint(
        query_molecule
    )

    df = pd.read_csv(DATA_PATH)

    if "mol" not in df.columns:
        return {
            "available": False,
            "reason": "Dataset does not contain a 'mol' column.",
        }

    dataset_fingerprints = []

    for dataset_smiles in df["mol"].dropna():
        molecule = Chem.MolFromSmiles(str(dataset_smiles))

        if molecule is None:
            continue

        fingerprint = FINGERPRINT_GENERATOR.GetFingerprint(
            molecule
        )

        dataset_fingerprints.append(fingerprint)

    if not dataset_fingerprints:
        return {
            "available": False,
            "reason": "No valid molecules found in the dataset.",
        }

    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp,
        dataset_fingerprints,
    )

    best_similarity = max(similarities)

    return {
        "available": True,
        "nearest_similarity": float(best_similarity),
        "compared_molecules": len(dataset_fingerprints),
    }


def assess_uncertainty(smiles):
    model = load_model()
    features = prepare_features(smiles)

    expected_features = getattr(
        model,
        "n_features_in_",
        features.shape[1],
    )

    if features.shape[1] != expected_features:
        raise ValueError(
            "Feature count does not match the saved model."
        )

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    active_probability = float(
        probabilities[classes.index(1)]
    )

    inactive_probability = float(
        probabilities[classes.index(0)]
    )

    predicted_class = int(model.predict(features)[0])

    entropy = calculate_entropy(probabilities)

    margin = abs(
        active_probability - inactive_probability
    )

    tree_disagreement = get_tree_disagreement(
        model,
        features,
    )

    chemical_similarity = calculate_nearest_similarity(
        smiles
    )

    # Simple interpretation rules, not calibrated confidence thresholds.
    if entropy >= 0.80:
        uncertainty_label = "High probability uncertainty"
    elif entropy >= 0.50:
        uncertainty_label = "Moderate probability uncertainty"
    else:
        uncertainty_label = "Low probability uncertainty"

    if chemical_similarity["available"]:
        similarity = chemical_similarity["nearest_similarity"]

        if similarity < 0.30:
            chemical_space_status = "Low similarity: possible out-of-domain molecule"
        elif similarity < 0.60:
            chemical_space_status = "Moderate similarity to dataset"
        else:
            chemical_space_status = "High similarity to dataset"
    else:
        chemical_space_status = "Not assessed"

    return {
        "smiles": smiles,
        "target": "BACE1",
        "predicted_class": predicted_class,
        "active_probability": active_probability,
        "inactive_probability": inactive_probability,
        "probability_entropy": entropy,
        "probability_margin": float(margin),
        "uncertainty_interpretation": uncertainty_label,
        "random_forest_tree_disagreement": tree_disagreement,
        "chemical_similarity": chemical_similarity,
        "chemical_space_status": chemical_space_status,
        "scientific_note": (
            "These are model reliability indicators, not calibrated "
            "probabilities of biological activity. Similarity to the "
            "dataset does not prove biological validity. Experimental "
            "validation is required."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assess MoleculeAI prediction uncertainty."
    )

    parser.add_argument(
        "--smiles",
        required=True,
        help="Molecule SMILES string.",
    )

    args = parser.parse_args()

    result = assess_uncertainty(args.smiles)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "uncertainty_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nMOLECULEAI UNCERTAINTY REPORT")
    print("=" * 50)
    print(f"SMILES: {result['smiles']}")
    print(f"Active probability: {result['active_probability']:.4f}")
    print(f"Inactive probability: {result['inactive_probability']:.4f}")
    print(f"Entropy: {result['probability_entropy']:.4f}")
    print(f"Margin: {result['probability_margin']:.4f}")
    print(f"Uncertainty: {result['uncertainty_interpretation']}")

    print("\nTree disagreement:")
    print(
        json.dumps(
            result["random_forest_tree_disagreement"],
            indent=2,
        )
    )

    print("\nChemical-space assessment:")
    print(result["chemical_space_status"])

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()