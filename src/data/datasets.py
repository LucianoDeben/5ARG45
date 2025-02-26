from data.loaders import GCTXDataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, Optional, List, Callable

class MultimodalDataset(Dataset):
    def __init__(
        self,
        gctx_loader: GCTXDataLoader,
        smiles_column: str,
        target_column: str,
        metadata_columns: Optional[List[str]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        chemical_descriptor_fn: Optional[Callable] = None,
        indices: Optional[List[int]] = None
    ):
        self.gctx_loader = gctx_loader
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.metadata_columns = metadata_columns or []
        self.transforms = transforms or {}
        self.indices = indices
        self.chemical_descriptor_fn = chemical_descriptor_fn or self._default_chemical_descriptor
        self._load_data()

    def _default_chemical_descriptor(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp, dtype=np.float32)

    def _load_data(self):
        with self.gctx_loader:
            row_slice = slice(None) if self.indices is None else self.indices
            self.transcriptomics = pd.DataFrame(self.gctx_loader.get_expression_data(row_slice=row_slice))
            row_metadata = self.gctx_loader.get_row_metadata(row_slice)

            for col in [self.smiles_column, self.target_column] + self.metadata_columns:
                if col not in row_metadata.columns:
                    raise KeyError(f"Column '{col}' not found in row metadata.")

            self.smiles = row_metadata[self.smiles_column].values
            self.targets = row_metadata[self.target_column].values.astype(np.float32)
            self.metadata = row_metadata[self.metadata_columns] if self.metadata_columns else None
            self.chemicals = self._compute_chemical_descriptors()

    def _compute_chemical_descriptors(self) -> pd.DataFrame:
        descriptors = [self.chemical_descriptor_fn(smiles) for smiles in self.smiles]
        return pd.DataFrame(descriptors, index=self.transcriptomics.index)

    def __len__(self) -> int:
        return len(self.transcriptomics)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        transcriptomic_data = self.transcriptomics.iloc[idx].values
        chemical_data = self.chemicals.iloc[idx].values
        target = self.targets[idx]

        if 'transcriptomics' in self.transforms:
            transcriptomic_data = self.transforms['transcriptomics'](transcriptomic_data)
        if 'chemicals' in self.transforms:
            chemical_data = self.transforms['chemicals'](chemical_data)

        sample = {
            'transcriptomics': torch.tensor(transcriptomic_data, dtype=torch.float32),
            'chemicals': torch.tensor(chemical_data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

        if self.metadata is not None:
            sample['metadata'] = self.metadata.iloc[idx].to_dict()

        return sample

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.hstack((self.transcriptomics.values, self.chemicals.values))
        y = self.targets
        return X, y

    def to_dataframe(self) -> tuple[pd.DataFrame, pd.Series]:
        features = pd.concat([self.transcriptomics, self.chemicals], axis=1)
        targets = pd.Series(self.targets, index=self.transcriptomics.index)
        return features, targets