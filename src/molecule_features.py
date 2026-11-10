import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

FINGERPRINT_SIZE = 2048


def _molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError("Invalid SMILES representation")

    return molecule


def _fingerprint(molecule):
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=2,
        nBits=FINGERPRINT_SIZE
    )
    return np.asarray(list(fingerprint), dtype=np.uint8)


def smiles_to_features(smiles):
    return _fingerprint(_molecule_from_smiles(smiles))


def batch_features(smiles_list):
    if not smiles_list:
        return np.empty((0, FINGERPRINT_SIZE), dtype=np.uint8)

    return np.vstack([smiles_to_features(smiles) for smiles in smiles_list])


def analyze_molecule(smiles):

    molecule = _molecule_from_smiles(smiles)

    properties = {
        "molecular_weight": Descriptors.MolWt(molecule),
        "logP": Descriptors.MolLogP(molecule),
        "hydrogen_donors": Descriptors.NumHDonors(molecule),
        "hydrogen_acceptors": Descriptors.NumHAcceptors(molecule),
        "rotatable_bonds": Descriptors.NumRotatableBonds(molecule)
    }

    fingerprint = _fingerprint(molecule)

    return properties, fingerprint.tolist()


if __name__ == "__main__":

    smiles = "CCO"

    properties, fingerprint = analyze_molecule(smiles)

    print("Molecular properties:")
    for key, value in properties.items():
        print(f"{key}: {value}")

    print("\nFingerprint length:", len(fingerprint))