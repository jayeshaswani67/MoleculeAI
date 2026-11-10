from pathlib import Path
import json
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from src.molecule_features import smiles_to_features
from src.explainability import explain_molecule
from src.uncertainty import assess_uncertainty
from src.benchmarking import run_benchmark

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "moleculeai_bace.joblib"
REPORT_DIR = ROOT / "reports"

app = Flask(__name__)


def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python -m src.train"
        )
    return joblib.load(MODEL_PATH)


def analyze_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string. Please check the molecular structure.")

    features = smiles_to_features(smiles)
    if features is None:
        raise ValueError("Could not generate molecular features.")

    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    model = get_model()

    expected = getattr(model, "n_features_in_", X.shape[1])
    if X.shape[1] != expected:
        raise ValueError(
            f"Feature mismatch: model expects {expected} features, "
            f"but the current feature extractor produced {X.shape[1]}."
        )

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    active_probability = float(probabilities[classes.index(1)])
    inactive_probability = float(probabilities[classes.index(0)])
    prediction = int(model.predict(X)[0])

    return {
        "smiles": smiles,
        "target": "BACE1",
        "prediction": prediction,
        "prediction_label": "Predicted active" if prediction == 1 else "Predicted inactive",
        "active_probability": active_probability,
        "inactive_probability": inactive_probability,
        "descriptors": {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "h_bond_donors": int(Lipinski.NumHDonors(mol)),
            "h_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        },
        "scientific_note": (
            "This is a computational prediction of BACE1 bioactivity, "
            "not evidence of drug efficacy, safety, or clinical benefit."
        ),
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "target": "BACE1",
    })


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    smiles = (payload.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "Please enter a SMILES string."}), 400
    try:
        return jsonify(analyze_smiles(smiles))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/explain")
def api_explain():
    payload = request.get_json(silent=True) or {}
    smiles = (payload.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "Please enter a SMILES string."}), 400
    try:
        result = explain_molecule(smiles, top_k=12, use_shap=True)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/uncertainty")
def api_uncertainty():
    payload = request.get_json(silent=True) or {}
    smiles = (payload.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "Please enter a SMILES string."}), 400
    try:
        return jsonify(assess_uncertainty(smiles))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/benchmark")
def api_benchmark():
    try:
        run_benchmark()
        csv_path = REPORT_DIR / "benchmark_results.csv"
        if not csv_path.exists():
            return jsonify({"error": "Benchmark finished but no CSV report was created."}), 500
        df = pd.read_csv(csv_path).replace({np.nan: None})
        return jsonify({"results": df.to_dict(orient="records")})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found."}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
