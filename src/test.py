import decoupler as dc
from data_sets import LINCSDataset

# Instantiate the dataset (assume you have the .gctx and everything else set up)
dataset = LINCSDataset(
    gctx_path="../data/processed/LINCS.gctx",
    normalize="z-score",
    feature_space="landmark"
)

# Suppose you have a TF network, e.g., CollecTRI or DoRothEA
net = dc.get_collectri(organism="human")

# Run TF inference using multiple methods, with a consensus
dataset.run_tf_interference(net=net, methods=["ulm", "viper"], consensus=True)

print(dataset.adata.obsm["consensus_estimate"])
