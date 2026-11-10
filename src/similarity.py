
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.data_loader import load_dataset


generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def find_similar_molecules(query_smiles, top_k=5):

    query = Chem.MolFromSmiles(query_smiles)

    if query is None:
        raise ValueError("Invalid molecular structure")

    query_fp = generator.GetFingerprint(query)
    query_canonical = Chem.MolToSmiles(query)

    df = load_dataset()

    results = []

    for _, row in df.iterrows():

        molecule = Chem.MolFromSmiles(row["mol"])

        if molecule is None:
            continue

        canonical = Chem.MolToSmiles(molecule)

        if canonical == query_canonical:
            continue

        fingerprint = generator.GetFingerprint(molecule)

        similarity = DataStructs.TanimotoSimilarity(
            query_fp,
            fingerprint
        )

        results.append({
            "smiles": row["mol"],
            "similarity": round(similarity, 4),
            "experimental_class": int(row["Class"])
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    query = "CC(=O)OC1=CC=CC=C1C(=O)O"

    results = find_similar_molecules(query)

    print("\nSIMILAR MOLECULE RESEARCH")
    print("=" * 45)

    for index, item in enumerate(results, 1):

        print(f"\nCompound {index}")
        print("SMILES:", item["smiles"])
        print("Similarity:", item["similarity"])
        print("Experimental BACE class:",
              item["experimental_class"])
