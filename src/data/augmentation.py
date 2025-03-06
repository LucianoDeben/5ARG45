# # src/data/augmentation.py
# import logging
# from typing import Callable, Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# import torch_geometric
# from rdkit import Chem

# logger = logging.getLogger(__name__)


# class GaussianNoise:
#     """Add Gaussian noise to transcriptomics data."""

#     def __init__(self, mean: float = 0.0, std: float = 0.01):
#         self.mean = mean
#         self.std = std

#     def __call__(
#         self, x: Union[np.ndarray, torch.Tensor]
#     ) -> Union[np.ndarray, torch.Tensor]:
#         if isinstance(x, np.ndarray):
#             noise = np.random.normal(self.mean, self.std, x.shape)
#             return x + noise
#         elif isinstance(x, torch.Tensor):
#             noise = torch.normal(self.mean, self.std, x.shape)
#             return x + noise
#         else:
#             raise TypeError("Input must be numpy array or torch tensor")


# class FeatureDropout:
#     """Randomly drop features (genes) in transcriptomics data."""

#     def __init__(self, drop_prob: float = 0.1):
#         self.drop_prob = drop_prob

#     def __call__(
#         self, x: Union[np.ndarray, torch.Tensor]
#     ) -> Union[np.ndarray, torch.Tensor]:
#         mask = np.random.binomial(1, 1 - self.drop_prob, x.shape[-1])
#         if isinstance(x, np.ndarray):
#             return x * mask
#         elif isinstance(x, torch.Tensor):
#             return x * torch.tensor(mask, dtype=x.dtype, device=x.device)
#         else:
#             raise TypeError("Input must be numpy array or torch tensor")


# class SmilesEnumeration:
#     """Generate random SMILES enumerations for data augmentation."""

#     def __init__(self, max_enumerations: int = 5):
#         self.max_enumerations = max_enumerations

#     def __call__(self, smiles: str) -> str:
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 return smiles
#             return Chem.MolToSmiles(mol, doRandom=True)
#         except Exception as e:
#             logger.warning(f"Failed to enumerate SMILES: {e}")
#             return smiles


# class MolecularGraphPerturbation:
#     """Perturb molecular graphs by randomly dropping edges (for GNNs)."""

#     def __init__(self, drop_prob: float = 0.1):
#         self.drop_prob = drop_prob

#     def __call__(self, graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
#         from torch_geometric.data import Data

#         if not isinstance(graph, Data):
#             raise TypeError("Input must be a torch_geometric.Data object")

#         edge_index = graph.edge_index
#         num_edges = edge_index.size(1) // 2  # Assuming undirected graph
#         keep_mask = torch.rand(num_edges) > self.drop_prob
#         keep_indices = torch.nonzero(keep_mask).squeeze()

#         new_edge_index = edge_index[:, keep_indices]
#         new_edge_attr = (
#             graph.edge_attr[keep_indices] if graph.edge_attr is not None else None
#         )

#         return Data(
#             x=graph.x,
#             edge_index=new_edge_index,
#             edge_attr=new_edge_attr,
#             y=graph.y,
#         )


# def create_augmentations(
#     transcriptomics_augment_type: Optional[str] = None,
#     molecular_augment_type: Optional[str] = None,
#     **kwargs,
# ) -> Tuple[Optional[Callable], Optional[Callable]]:
#     """Factory function to create augmentation transforms."""
#     transcriptomics_augment = None
#     if transcriptomics_augment_type == "noise":
#         transcriptomics_augment = GaussianNoise(**kwargs.get("noise_args", {}))
#     elif transcriptomics_augment_type == "dropout":
#         transcriptomics_augment = FeatureDropout(**kwargs.get("dropout_args", {}))

#     molecular_augment = None
#     if molecular_augment_type == "smiles_enum":
#         molecular_augment = SmilesEnumeration(**kwargs.get("enum_args", {}))
#     elif molecular_augment_type == "graph_perturb":
#         molecular_augment = MolecularGraphPerturbation(**kwargs.get("perturb_args", {}))

#     return transcriptomics_augment, molecular_augment
