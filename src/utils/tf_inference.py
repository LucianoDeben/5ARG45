import logging
import numpy as np
import pandas as pd
from typing import Tuple, List
import decoupler as dc

logger = logging.getLogger(__name__)

def apply_tf_inference(
    data: np.ndarray, 
    gene_symbols: List[str], 
    network: pd.DataFrame,
    method: str = "ulm",
    min_n: int = 10,
    use_raw: bool = False,
    consensus: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply TF inference to gene expression data.
    
    Args:
        data: Gene expression matrix (n_samples x n_genes)
        gene_symbols: List of gene symbols corresponding to columns
        network: Regulatory network for TF inference
        method: Inference method (or list of methods)
        min_n: Minimum number of targets per TF
        use_raw: Whether to use raw values
        consensus: Whether to use consensus of methods
        
    Returns:
        Tuple of (transformed_data, feature_names)
    """
    logger.info(f"Applying TF inference to data with {len(gene_symbols)} genes")
    
    # Create a DataFrame for decoupler
    X_df = pd.DataFrame(data, columns=gene_symbols)
    
    try:
        # Convert method to list if string
        methods = [method] if isinstance(method, str) else method
        
        # Run decoupler
        res = dc.decouple(
            X_df,
            network,
            methods=methods,
            consensus=consensus,
            min_n=min_n, 
            use_raw=use_raw
        )
        
        # Get TF activity scores
        tf_activities, _ = dc.cons(res)
        
        # Get TF names
        tf_names = list(tf_activities.columns)
        
        logger.info(f"TF inference completed: reduced from {data.shape[1]} genes to {len(tf_names)} TFs")
        
        return tf_activities.values, tf_names
        
    except Exception as e:
        logger.error(f"Error in TF inference: {str(e)}")
        logger.warning("Falling back to original gene expression data")
        return data, gene_symbols