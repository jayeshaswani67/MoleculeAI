# MoleculeAI Flask interface

This adds a lightweight Flask web interface to the existing MoleculeAI project.

## Expected project layout

```text
MoleculeAI/
├── app.py
├── src/
│   ├── molecule_features.py
│   ├── explainability.py
│   ├── uncertainty.py
│   └── benchmarking.py
├── models/
│   └── moleculeai_bace.joblib
├── data/
│   └── raw/
│       └── bace.csv
├── reports/
├── templates/
│   └── index.html
└── static/
    └── app.js
```

Copy `app.py` into the project root. Copy the `templates` and `static` folders into the same root.

## Install Flask

In Windows CMD:

```cmd
cd /d E:\MoleculeAI
venv\Scripts\activate
pip install flask
```

Add this line to your existing `requirements.txt`:

```text
flask
```

## Run

Make sure the trained model exists first:

```cmd
python -m src.train
```

Then start the interface:

```cmd
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

## API endpoints

- `GET /api/health`
- `POST /api/analyze` with JSON `{"smiles":"CCO"}`
- `POST /api/explain` with JSON `{"smiles":"CCO"}`
- `POST /api/uncertainty` with JSON `{"smiles":"CCO"}`
- `POST /api/benchmark` with JSON `{}`

The interface and API are served by the same Flask process, so CORS configuration is not required.

## Important integration checks

The backend expects these functions to exist:

- `src.molecule_features.smiles_to_features`
- `src.explainability.explain_molecule`
- `src.uncertainty.assess_uncertainty`
- `src.benchmarking.run_benchmark`

It also expects the saved model at `models/moleculeai_bace.joblib`.

The model feature order and feature count must match the saved model. The UI's probability display is the model's class score, not a calibrated probability of clinical success.

This is a research prototype for BACE1 bioactivity, not a medical or clinical decision system.
