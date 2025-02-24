import inspect
import pandas as pd
import numpy as np
import scanpy as sc
import decoupler as dc
from typing import List, Optional, Union

class TFInferenceRunner:
    """
    A utility class to infer transcription factor (TF) activities from gene expression data using decoupler.

    This class processes a gene expression matrix and a TF network to estimate TF activities without modifying
    the input data. It supports multiple inference methods, optional consensus scoring, and flexible configuration.

    Parameters:
        methods (List[str]): List of decoupler methods to use (e.g., ["ulm", "mlm"]). Default: ["ulm"].
        consensus (bool): If True and multiple methods are provided, compute a consensus score. Default: False.
        consensus_key (str): Name of the consensus output column. Default: "consensus".
        source_col (str): Column name in the network DataFrame for TF sources. Default: "source".
        target_col (str): Column name in the network DataFrame for target genes. Default: "target".
        weight_col (str): Column name in the network DataFrame for interaction weights. Default: "weight".
        verbose (bool): If True, print progress messages. Default: True.
        min_n (int): Minimum number of targets per source for inference. Default: 1.
        **method_kwargs: Additional keyword arguments passed to decoupler methods.
    """
    def __init__(
        self,
        methods: List[str] = ["ulm"],
        consensus: bool = False,
        consensus_key: str = "consensus",
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str = "weight",
        verbose: bool = True,
        min_n: int = 1,
        **method_kwargs
    ):
        self.methods = methods
        self.consensus = consensus
        self.consensus_key = consensus_key
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.verbose = verbose
        self.min_n = min_n
        self.method_kwargs = method_kwargs

    def run(
        self,
        expr_data: Union[np.ndarray, pd.DataFrame],
        net: pd.DataFrame,
        row_ids: Optional[List[str]] = None,
        col_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Infers TF activities from gene expression data and a TF network.

        Args:
            expr_data (Union[np.ndarray, pd.DataFrame]): Gene expression matrix (n_samples x n_genes).
                If NumPy array, col_ids must be provided to map columns to gene symbols.
                If DataFrame, column names are used as gene IDs.
            net (pd.DataFrame): TF network with source, target, and weight columns.
            row_ids (Optional[List[str]]): List of sample IDs to attach to the output. Default: None.
            col_ids (Optional[List[str]]): List of gene symbols corresponding to expr_data columns.
                Required if expr_data is a NumPy array. Default: None.

        Returns:
            pd.DataFrame: Inferred TF activity matrix with rows as samples and columns as TFs.
                If consensus=True, returns only the consensus scores. If no estimates are produced,
                returns an empty DataFrame.
        """
        # Convert expr_data to DataFrame with appropriate column names
        if isinstance(expr_data, np.ndarray):
            if col_ids is None:
                raise ValueError("col_ids must be provided when expr_data is a NumPy array!")
            if len(col_ids) != expr_data.shape[1]:
                raise ValueError(f"col_ids length ({len(col_ids)}) does not match expr_data columns ({expr_data.shape[1]})!")
            if row_ids is None:
                row_ids = [f"sample_{i}" for i in range(expr_data.shape[0])]
            expr_df = pd.DataFrame(expr_data, index=row_ids, columns=col_ids)
        else:
            expr_df = expr_data.copy()
            row_ids = expr_df.index.tolist() if row_ids is None else row_ids

        # Filter to shared genes between expr_data and net
        shared_genes = set(net[self.target_col]).intersection(expr_df.columns)
        if not shared_genes:
            raise ValueError("No shared genes found between expression data and network!")
        filtered_columns = [col for col in expr_df.columns if col in shared_genes]
        expr_df_filtered = expr_df[filtered_columns]
        
        # Filter the network to shared genes
        net_filtered = net[net[self.target_col].isin(shared_genes)]
        if self.verbose:
            print(f"Filtered network has {len(net_filtered)} interactions on {len(shared_genes)} shared genes.")

        # Create an AnnData object
        adata = sc.AnnData(
            X=expr_df_filtered.values,
            obs=pd.DataFrame(index=expr_df_filtered.index),
            var=pd.DataFrame(index=expr_df_filtered.columns)
        )
        
        # Run each specified method
        results = {}
        for method in self.methods:
            method_fn = getattr(dc, f"run_{method}", None)
            if method_fn is None:
                raise ValueError(f"Method '{method}' not found in decoupler!")
            # Build kwargs, adding "weight" only if the function accepts it
            kwargs = {"min_n": self.min_n, "verbose": self.verbose}
            sig = inspect.signature(method_fn)
            if "weight" in sig.parameters:
                kwargs["weight"] = self.weight_col
            kwargs.update(self.method_kwargs)
            if self.verbose:
                print(f"Running TF inference with method: {method}")
            method_fn(
                mat=adata,
                net=net_filtered,
                source=self.source_col,
                target=self.target_col,
                use_raw=False,
                **kwargs
            )
            obsm_key = f"{method.lower()}_estimate"
            if obsm_key in adata.obsm:
                results[method] = adata.obsm[obsm_key]
            else:
                if self.verbose:
                    print(f"Warning: Method {method} did not produce an estimate.")

        # Handle output
        if not results:
            if self.verbose:
                print("No estimates produced by any method. Returning empty DataFrame.")
            return pd.DataFrame(index=row_ids)

        if self.consensus and len(self.methods) > 1:
            if self.verbose:
                print("Computing consensus across methods...")
            consensus_est, _ = dc.cons(results)
            result_df = pd.DataFrame(consensus_est, index=row_ids, columns=results[self.methods[0]].columns)
            result_df.columns = [f"{self.consensus_key}_{col}" for col in result_df.columns]
        else:
            method = self.methods[0]
            result_df = pd.DataFrame(results[method], index=row_ids, columns=results[method].columns)

        result_df.fillna(0.0, inplace=True)
        if self.verbose:
            print("TF inference complete.")
        return result_df

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(methods={self.methods}, "
                f"consensus={self.consensus}, verbose={self.verbose})")