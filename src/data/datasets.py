import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SMILESTokenizer:
    def __init__(self):
        standard_smiles_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-=#%()[]{}@+*/\.,!?$%^&|~"
        self.vocab = standard_smiles_chars
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def tokenize(self, smiles: str) -> List[int]:
        return [self.char_to_index[char] for char in smiles if char in self.vocab]

    def detokenize(self, tokens: List[int]) -> str:
        return "".join(self.index_to_char[token] for token in tokens)


class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for multimodal drug response prediction with LINCS/CTRP data.

    Args:
        transcriptomics: Preloaded gene expression DataFrame.
        row_metadata: Preloaded row metadata DataFrame.
        smiles_column: Column name for SMILES strings.
        target_column: Column name for target values.
        metadata_columns: Optional metadata columns.
        transforms: Optional preprocessing transforms.
        chemical_descriptor_fn: Custom descriptor function (default: ECFP).
        chemical_representation: 'fingerprint' (ECFP) or 'smiles_sequence'.
    """

    def __init__(
        self,
        transcriptomics: pd.DataFrame,
        row_metadata: pd.DataFrame,
        smiles_column: str = "canonical_smiles",
        target_column: str = "viability",
        metadata_columns: Optional[List[str]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        chemical_descriptor_fn: Optional[Callable] = None,
        chemical_representation: str = "fingerprint",
    ):
        self.transcriptomics = transcriptomics
        self.row_metadata = row_metadata
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.metadata_columns = metadata_columns or []
        self.transforms = transforms or {}
        self.chemical_descriptor_fn = (
            chemical_descriptor_fn or self._default_chemical_descriptor
        )
        self.chemical_representation = chemical_representation

        if self.chemical_representation == "smiles_sequence":
            self.tokenizer = SMILESTokenizer()
        else:
            self.tokenizer = None

        self._load_data()

    def _default_chemical_descriptor(self, smiles: str) -> np.ndarray:
        """Compute ECFP4 fingerprints (radius=2, 1024 bits) using MorganGenerator."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)  # ECFP4 equivalent
        fp = gen.GetFingerprint(mol)  # Returns ExplicitBitVect
        return np.frombuffer(fp.ToBinary(), dtype=np.uint8).astype(np.float32)

    def _compute_chemical_descriptors(self) -> pd.DataFrame:
        """Compute chemical descriptors for all SMILES strings in parallel."""
        descriptors = Parallel(n_jobs=-1)(
            delayed(self.chemical_descriptor_fn)(smiles) for smiles in self.smiles
        )
        return pd.DataFrame(descriptors, index=self.transcriptomics.index)

    def _load_data(self):
        """Validate and process preloaded data."""
        for col in [self.smiles_column, self.target_column] + self.metadata_columns:
            if col not in self.row_metadata.columns:
                raise KeyError(f"Column '{col}' not found in row metadata.")

        self.smiles = self.row_metadata[self.smiles_column].values
        self.targets = self.row_metadata[self.target_column].values.astype(np.float32)
        self.metadata = (
            self.row_metadata[self.metadata_columns] if self.metadata_columns else None
        )

        if self.chemical_representation == "fingerprint":
            self.chemicals = self._compute_chemical_descriptors()
        else:
            self.chemicals = None

        logger.debug(
            f"Loaded dataset with {len(self.transcriptomics)} samples, "
            f"{len(self.transcriptomics.columns)} genes, "
            f"{'ECFP fingerprints' if self.chemical_representation == 'fingerprint' else 'SMILES sequences'}"
        )

    def __len__(self) -> int:
        return len(self.transcriptomics)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict, List[int]]]:
        transcriptomic_data = self.transcriptomics.iloc[idx].values
        target = self.targets[idx]

        if self.chemical_representation == "fingerprint":
            chemical_data = self.chemicals.iloc[idx].values
        elif self.chemical_representation == "smiles_sequence":
            smiles = self.smiles[idx]
            chemical_data = self.tokenizer.tokenize(smiles)
        else:
            raise ValueError(
                f"Invalid chemical_representation: {self.chemical_representation}"
            )

        if "transcriptomics" in self.transforms:
            transcriptomic_data = self.transforms["transcriptomics"](
                transcriptomic_data
            )
        if (
            self.chemical_representation == "fingerprint"
            and "chemicals" in self.transforms
        ):
            chemical_data = self.transforms["chemicals"](chemical_data)

        sample = {
            "transcriptomics": torch.tensor(transcriptomic_data, dtype=torch.float32),
            "chemicals": (
                torch.tensor(chemical_data, dtype=torch.float32)
                if self.chemical_representation == "fingerprint"
                else chemical_data
            ),
            "target": torch.tensor(target, dtype=torch.float32),
        }

        if self.metadata is not None:
            sample["metadata"] = self.metadata.iloc[idx].to_dict()

        return sample

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.chemical_representation != "fingerprint":
            raise ValueError("to_numpy only supported for 'fingerprint' representation")
        X = np.hstack((self.transcriptomics.values, self.chemicals.values))
        y = self.targets
        return X, y

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.chemical_representation != "fingerprint":
            raise ValueError(
                "to_dataframe only supported for 'fingerprint' representation"
            )
        features = pd.concat([self.transcriptomics, self.chemicals], axis=1)
        targets = pd.Series(self.targets, index=self.transcriptomics.index)
        return features, targets
