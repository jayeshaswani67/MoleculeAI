
import joblib

from pathlib import Path

from src.molecule_features import smiles_to_features


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "moleculeai_bace.joblib"


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    return joblib.load(MODEL_PATH)


def predict_activity(smiles):

    model = load_model()

    features = smiles_to_features(smiles)
    features = features.reshape(1, -1)

    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    active_probability = float(
        probabilities[classes.index(1)]
    )

    result = {
        "smiles": smiles,
        "predicted_class": prediction,
        "active_class_probability": active_probability,
        "interpretation": (
            "Predicted active"
            if prediction == 1
            else "Predicted inactive"
        )
    }

    return result


if __name__ == "__main__":

    test_smiles = "CCO"

    try:
        result = predict_activity(test_smiles)

        print("\nMOLECULEAI PREDICTION")

        for key, value in result.items():
            print(f"{key}: {value}")

        print(
            "\nNote: This is a computational prediction, "
            "not evidence of drug efficacy or safety."
        )

    except Exception as error:
        print("Prediction error:", error)
