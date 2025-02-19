import decoupler as dc
from data_sets import LINCSDataset

from preprocess import run_tf_activity_inference
from utils import load_sampled_data

# # Load Collectri network
# collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# # Load the sampled data
# data = load_sampled_data(file_path="../data/processed/preprocessed_landmark.csv", sample_size=1000)

# tf_data = run_tf_activity_inference(data, collectri_net, min_n=1, algorithm="ulm")


# print(tf_data)



# Example: Single interference method
dataset = LINCSDataset(
    gctx_path="../data/processed/LINCS.gctx",
    normalize="z-score",
    feature_space="landmark"
)
# Get a TF network, e.g., CollecTRI from decoupler
net = dc.get_collectri(organism="human")

# Run TF interference using a single method ('ulm') without consensus.
tf_result_single = dataset.run_tf_interference(
    net=net,
    methods=["ulm"],
    consensus=False, # Not computing consensus when using a single method
    min_n = 1
)
print("Single interference result:")
print(tf_result_single)


# Run TF interference using multiple methods (e.g., 'ulm' and 'viper') and compute the consensus.
tf_result_consensus = dataset.run_tf_interference(
    net=net,
    methods=["ulm", "viper"],
    consensus=True,  # This will return only the consensus TF matrix
    min_n = 1
)
print("Consensus TF interference result:")
print(tf_result_consensus)
