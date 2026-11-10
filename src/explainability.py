from pathlib import Path
import argparse
import json

import joblib
import numpy as np
from rdkit import Chem

from src.molecule_features import smiles_to_features


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "moleculeai_bace.joblib"
REPORT_DIR = ROOT / "reports"

FINGERPRINT_SIZE = 2048


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train your model first using: python -m src.train"
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
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The loaded model must support predict_proba()."
        )

    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    active_index = classes.index(1)

    return float(probabilities[active_index])


def explain_with_perturbation(model, features, top_k=15):
    """
    Remove one active feature at a time and observe the score change.

    Positive impact: removing the feature lowers the active score.
    Negative impact: removing the feature raises the active score.

    This is a sensitivity analysis, not a causal explanation.
    """
    original_score = get_active_probability(model, features)

    vector = features[0]
    active_indices = np.flatnonzero(vector)

    impacts = []

    for index in active_indices:
        modified = features.copy()
        modified[0, index] = 0.0

        modified_score = get_active_probability(
            model,
            modified,
        )

        impact = original_score - modified_score

        if index < FINGERPRINT_SIZE:
            feature_name = f"Morgan fingerprint bit {index}"
        else:
            feature_name = f"Descriptor feature {index - FINGERPRINT_SIZE}"

        impacts.append({
            "feature_index": int(index),
            "feature_name": feature_name,
            "impact": float(impact),
            "score_after_removal": float(modified_score),
        })

    impacts.sort(
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )

    return {
        "method": "single_feature_perturbation",
        "baseline_active_score": original_score,
        "features": impacts[:top_k],
    }


def explain_with_shap(model, features, top_k=15):
    """Use SHAP TreeExplainer when available and compatible."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        # SHAP may return a list for classifiers or an array.
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])[0]
        else:
            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0, :, 1]
            elif values.ndim == 2:
                values = values[0]

        values = values.reshape(-1)

        impacts = []

        for index, value in enumerate(values):
            if index < FINGERPRINT_SIZE:
                feature_name = f"Morgan fingerprint bit {index}"
            else:
                feature_name = (
                    f"Descriptor feature {index - FINGERPRINT_SIZE}"
                )

            impacts.append({
                "feature_index": int(index),
                "feature_name": feature_name,
                "impact": float(value),
            })

        impacts.sort(
            key=lambda item: abs(item["impact"]),
            reverse=True,
        )

        return {
            "method": "shap",
            "features": impacts[:top_k],
        }

    except ImportError:
        return None

    except Exception as error:
        print(f"SHAP unavailable for this model: {error}")
        return None


def explain_molecule(smiles, top_k=15, use_shap=True):
    model = load_model()
    features = prepare_features(smiles)

    if features.shape[1] != getattr(model, "n_features_in_", features.shape[1]):
        raise ValueError(
            "Feature count does not match the trained model. "
            "Check molecule_features.py and the saved model."
        )

    prediction = int(model.predict(features)[0])
    active_score = get_active_probability(model, features)

    explanation = None

    if use_shap:
        explanation = explain_with_shap(
            model,
            features,
            top_k,
        )

    if explanation is None:
        explanation = explain_with_perturbation(
            model,
            features,
            top_k,
        )

    return {
        "smiles": smiles,
        "target": "BACE1",
        "predicted_class": prediction,
        "interpretation": (
            "Predicted active for BACE1"
            if prediction == 1
            else "Predicted inactive for BACE1"
        ),
        "active_class_score": active_score,
        "explanation": explanation,
        "scientific_note": (
            "Feature explanations describe model sensitivity or "
            "attribution. They do not prove biological causation."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Explain a MoleculeAI BACE1 prediction."
    )

    parser.add_argument(
        "--smiles",
        required=True,
        help="Molecule SMILES string.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of influential features to display.",
    )

    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Use perturbation instead of trying SHAP.",
    )

    args = parser.parse_args()

    result = explain_molecule(
        args.smiles,
        top_k=args.top_k,
        use_shap=not args.no_shap,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "molecule_explanation.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nMOLECULEAI EXPLAINABILITY REPORT")
    print("=" * 50)
    print(f"SMILES: {result['smiles']}")
    print(f"Prediction: {result['interpretation']}")
    print(f"Active-class score: {result['active_class_score']:.4f}")
    print(f"Explanation method: {result['explanation']['method']}")

    print("\nMost influential features:")

    for item in result["explanation"]["features"]:
        print(
            f"{item['feature_name']}: "
            f"{item['impact']:+.5f}"
        )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()