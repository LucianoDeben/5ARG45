import logging
from src.config.config_utils import setup_logging
from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXDataLoader
from torch.utils.data import DataLoader

logger = setup_logging()

lincs = GCTXDataLoader("./data/processed/LINCS_CTRP_matched.gctx")
mixseq = GCTXDataLoader("./data/processed/MixSeq.gctx")

print(mixseq.get_column_metadata_keys())
print(mixseq.get_row_metadata_keys())

# Load the MixSeq datasets
train_ds_mixseq, val_ds_mixseq, test_ds_mixseq = DatasetFactory.create_and_split_datasets(
    gctx_loader=mixseq,
    dataset_type="transcriptomics",
    feature_space="landmark",
    nrows=None,
    test_size=0.4,
    val_size=0.1,
    random_state=50,
    chunk_size=10000,
    group_by=None,
    stratify_by=None,
)

# Load the LINCS datasets
train_ds_lincs, val_ds_lincs, test_ds_lincs = DatasetFactory.create_and_split_datasets(
    gctx_loader=lincs,
    dataset_type="transcriptomics",
    feature_space="landmark",
    nrows=None,
    test_size=0.4,
    val_size=0.1,
    random_state=50,
    chunk_size=10000,
    group_by=None,
    stratify_by=None,
)

# Create all the DataLoaders
train_dl_mixseq = DataLoader(train_ds_mixseq, batch_size=32, shuffle=True)
val_dl_mixseq = DataLoader(val_ds_mixseq, batch_size=32, shuffle=False)
test_dl_mixseq = DataLoader(test_ds_mixseq, batch_size=32, shuffle=False)

# Create all the DataLoaders
train_dl_lincs = DataLoader(train_ds_lincs, batch_size=32, shuffle=True)
val_dl_lincs = DataLoader(val_ds_lincs, batch_size=32, shuffle=False)
test_dl_lincs = DataLoader(test_ds_lincs, batch_size=32, shuffle=False)



