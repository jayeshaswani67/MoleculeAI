
# MoleculeAI 🧬

### AI-Powered Molecular Intelligence & Drug Discovery Research

MoleculeAI is an experimental machine learning project designed
to help researchers analyze chemical molecules, explore molecular
similarities, and investigate target-specific biological activity.

## Current Capabilities

- Molecular structure processing using RDKit.
- SMILES-based molecular analysis.
- Molecular descriptors and Morgan fingerprints.
- BACE1 bioactivity classification.
- Random Forest machine learning.
- Chemical scaffold-based dataset splitting.
- Model evaluation and prediction.
- Molecular similarity search using Tanimoto similarity.

## Technology Stack

- Python
- RDKit
- Scikit-learn
- Pandas
- NumPy
- Random Forest
- MoleculeNet / BACE dataset

## Project Structure

```text
MoleculeAI/
├── data/
├── models/
├── src/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd MoleculeAI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Analyze a molecule:

```bash
python main.py analyze --smiles "CCO"
```

Train the model:

```bash
python -m src.train
```

Predict biological activity:

```bash
python main.py predict --smiles "CCO"
```

Find similar molecules:

```bash
python -m src.similarity
```

Test model performance:

```bash
python -m src.test_model
```

## Dataset

The initial model uses the BACE bioactivity dataset.

The dataset is downloaded automatically by the data loader.

## Research Limitations

MoleculeAI is an experimental computational research prototype.

Its predictions:
- Are specific to the BACE bioactivity classification task.
- Do not establish clinical efficacy.
- Do not establish drug safety.
- Require independent experimental validation.
- Should not be used for medical decisions.

## Future Development

- Graph Neural Networks.
- Protein structure analysis.
- Molecular docking.
- ADMET prediction.
- Explainable molecular AI.
- Generative molecular modelling.
- Multi-target bioactivity prediction.

## Project Status

Early-stage research prototype — Version 1.0.

## License

MIT License (subject to review of third-party dataset and dependency terms).
