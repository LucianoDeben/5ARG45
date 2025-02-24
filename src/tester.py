import pandas as pd
from data_sets import LINCSDataset
from tf_inference import TFInferenceRunner
import decoupler as dc


dataset = LINCSDataset(gctx_path="../data/processed/LINCS.gctx", normalize=None, feature_space="landmark", nrows=1000)
runner = TFInferenceRunner(methods=["mlm"], consensus=False, min_n=1)
net = dc.get_collectri(organism="human")

tf_activities = dataset.infer_tf_activities(net, runner)
print(tf_activities.head())