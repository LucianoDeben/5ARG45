import logging
import decoupler as dc
import pandas as pd
from data_sets import LINCSDataset

from preprocess import run_tf_activity_inference
from utils import load_sampled_data

# # Load Collectri network
# collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# # Load the sampled data
# data = load_sampled_data(file_path="../data/processed/preprocessed_landmark.csv", sample_size=1000)


# dataset = LINCSDataset(
#     gctx_path="../data/processed/LINCS.gctx",
#     normalize="z-score",
#     feature_space="landmark"
# )

# matrix = dataset.get_expression_matrix()

# df = pd.DataFrame(matrix)


# tf_data = run_tf_activity_inference(df, collectri_net, min_n=1, algorithm="ulm")


# print(tf_data)



# Example: Single interference method
dataset = LINCSDataset(
    gctx_path="../data/processed/LINCS.gctx",
    normalize="z-score",
    feature_space="landmark"
)

matrix = dataset.get_expression_matrix()
print("Expression matrix sample:")
logging.info(f"Sample data:\n{matrix[:20, :20]}")
# Get a TF network, e.g., CollecTRI from decoupler
net = dc.get_collectri(organism="human")

# # Run TF interference using a single method ('ulm') without consensus.
# tf_result_single = dataset.run_tf_interference(
#     net=net,
#     methods=["ulm"],
#     consensus=False, # Not computing consensus when using a single method
#     min_n = 1
# )
# print("Single interference result:")
# print(tf_result_single)


# Run TF interference using multiple methods (e.g., 'ulm' and 'viper') and compute the consensus.
tf_result_consensus = dataset.run_tf_interference(
    net=net,
    methods=["ulm", "viper"],
    consensus=True,  # This will return only the consensus TF matrix
    min_n = 1
)
print("Consensus TF interference result:")
print(tf_result_consensus)
