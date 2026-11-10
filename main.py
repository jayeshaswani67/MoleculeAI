
import argparse

from src.train import train
from src.predict import predict_activity
from src.molecule_features import analyze_molecule


def main():

    parser = argparse.ArgumentParser(
        description="MoleculeAI - AI Drug Discovery Research"
    )

    parser.add_argument(
        "action",
        choices=["train", "predict", "analyze"],
        help="Choose an operation"
    )

    parser.add_argument(
        "--smiles",
        type=str,
        help="Molecular SMILES string"
    )

    args = parser.parse_args()

    if args.action == "train":

        train()

    elif args.action == "analyze":

        if not args.smiles:
            parser.error("--smiles is required for analyze")

        properties, fingerprint = analyze_molecule(args.smiles)

        print("\nMOLECULAR ANALYSIS")

        for name, value in properties.items():
            print(f"{name}: {value}")

        print("Fingerprint length:", len(fingerprint))

    elif args.action == "predict":

        if not args.smiles:
            parser.error("--smiles is required for predict")

        result = predict_activity(args.smiles)

        print("\nMOLECULEAI RESULT")

        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
