from src.data.loaders import GCTXLoader


loader = GCTXLoader("./data/processed/LINCS_small_updated.gctx")

ids = loader.get_row_ids(row_indices=[0,1,2,3,4,5,6,7,8,9])
print("Number of Gene Symbols:", len(ids))
print("Gene Symbols:", ids)