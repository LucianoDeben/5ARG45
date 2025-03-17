# %%
import json
import h5py
import pickle as pk
import numpy as np
import pandas as pd
import decoupler as dc
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
import networkx as nx

# Load environment configuration
with open("environment.json") as handle:
    env = json.load(handle)
    print("environment", env['name'], "loaded")

# %%
# Loading in LINCS
geneinfo = pd.read_csv(env['geneinfo'], sep="\t")
geneinfo_df = geneinfo.set_index('gene_id')

# Fix the DtypeWarning with low_memory=False
siginfo = pd.read_csv(env['siginfo'], sep="\t", low_memory=False)

# Keep only 24 hour timepoint
siginfo_df = siginfo.loc[siginfo.pert_time == 24.0]

# Keep only signatures with filled in perturbation dosage
siginfo_df = siginfo_df.loc[~(siginfo_df.pert_dose.isna())]

# Reset the index
siginfo_df.reset_index(drop=False, inplace=True)

# Loading in the .gctx file
f = h5py.File(env["gctx_file"])

# Signature ids (in gctx order)
sig_ids = pd.Series(f['0']['META']['COL']['id']).astype(str)

# Gene ids (in gctx order)
gene_ids = np.array(f['0']['META']['ROW']['id']).astype(int)

# Gene symbols (in gctx order)
gene_symbols = geneinfo_df.loc[gene_ids].gene_symbol.tolist()

# The matrix (gctx) indices corresponding to landmark genes
landmark_idxs = np.where(geneinfo_df.loc[gene_ids].feature_space.isin(["landmark"]))[0]

# The matrix object of gctx containing differential expression (signature) values
matrix = f['0']['DATA']['0']['matrix']

# Gene symbols list in order in which they are provided after indexing the gctx data
landmark_gene_symbol = geneinfo_df.loc[gene_ids].iloc[landmark_idxs].gene_symbol.tolist()
all_gene_symbol = geneinfo_df.loc[gene_ids].gene_symbol.tolist()

# Gene info in gxtx order
geneinfo_ordered = geneinfo_df.loc[gene_ids]

# Loading in CTRP
per_cpd_post_qc = pd.read_csv(env["per_cpd_post_qc"], sep="\t")
curves_df = pd.read_csv(env["curves_post_qc"], sep="\t")
per_compound = pd.read_csv(env["per_compound"], sep="\t")
per_cell_line = pd.read_csv(env["per_cell_line"], sep="\t")
per_experiment = pd.read_csv(env["per_experiment"], sep="\t")

# Loading in our GRN
collectri_net = dc.get_collectri()

# %%
# ======= Defining Functions ========

def sig3upper(p, X):
    """
    This is the curve implementation from the CTRP provided Matlab script
    with added numerical stability.
    """
    alpha = p[0]      # EC50
    beta = p[1]       # slope
    b = p[2]          # lower limit
    
    # Add numerical stability by clipping extreme values
    exponent = np.clip(-(X-alpha)/beta, -709, 709)  # prevent overflow
    A = b + (1-b) / (1 + np.exp(exponent))
    return A

def get_viability_mapping(ctrp_idx_mapping):
    """
    Returns a dictionary mapping siginfo index to a list of curve-evaluated viabilities.
    """
    viability_mapping = dict()
    for key, val in tqdm(list(ctrp_idx_mapping.items())):
        curves_i = curves_df.loc[val]
        x = siginfo_df.loc[key].pert_dose
        viability = []
        for i, curve in curves_i.iterrows():
            p = curve[['p1_center', 'p2_slope', 'p4_baseline']].tolist()
            eval = sig3upper(p, np.log2(x))
            viability.append(eval)
        viability_mapping[key] = viability
    return viability_mapping

def get_ctrp_idx_mapping(siginfo_in_isct, ctrp_in_isct, isct_list):
    """
    Returns a dictionary mapping siginfo index to ctrp index.
    """
    # map pair to ctrp index
    ctrp_in_isct_mapping = {i:ctrp_in_isct.index[ctrp_in_isct == i].tolist() for i in tqdm(isct_list)}

    # map siginfo index to ctrp index
    ctrp_idx_mapping = {i:ctrp_in_isct_mapping[pair] for i, pair in tqdm(siginfo_in_isct.items())}
    return ctrp_idx_mapping

def run_inference_on_slice(slice_idx):
    """
    Takes the numeric slice indices.
    returns a dataframe "acts" with as indices the slice sig_ids
    (This reduces order confounding as it enables reordering)
    """
    print("Indexing matrix")
    sample = pd.DataFrame(matrix[slice_idx.index,:].T
                        , index=gene_symbols
                        , columns=slice_idx.tolist()).T
    mia_mean = sample['MIA2'].mean(axis=1)
    sample.drop("MIA2", axis=1, inplace=True)
    sample["MIA2"] = mia_mean

    print("Inference")
    acts = dc.run_mlm(sample, net=collectri_net)
    return acts

def lookup_pair_per_cpd_post_qc(pair_experiment_id, pair_master_cpd_id):
    """
    For a given experiment id and compound, returns the per_cpd_post dataframe matching rows.
    """
    experiment_id_bool = (per_cpd_post_qc.experiment_id == pair_experiment_id)
    master_cpd_id_bool = (per_cpd_post_qc.master_cpd_id == pair_master_cpd_id)

    pair_bool = experiment_id_bool & master_cpd_id_bool
    pair_per_cpd_post_qc = per_cpd_post_qc.loc[pair_bool]
    return pair_per_cpd_post_qc

def get_min(ctrp_idx_first):
    """
    From a list of ctrp indices, fetches the lowest concentration value (log2)
    """
    min_list = list()

    for ctrp_i in tqdm(ctrp_idx_first):
        # Index the curve to acquire the experiemnt id and compound id
        curve = curves_df.loc[ctrp_i]

        # Get the cpd_conc_post_qc subset
        cpd_post_qc_subset = lookup_pair_per_cpd_post_qc(curve.experiment_id, curve.master_cpd_id)
        curve_conc = cpd_post_qc_subset.cpd_conc_umol

        # Note that we take the log2 of the concentration as per convention in handling these concentrations
        curve_conc_min = np.log2(curve_conc.min())

        min_list.append(curve_conc_min)
    return min_list

def convert_smile(smile):
    """
    Takes a smile, and uses the Chem package to convert it to a networkx graph.
    """
    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    # Convert RDKit Mol to NetworkX Graph
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())

    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=bond.GetBondType())

    return graph

def create_gctx_file(data_dict, output_path="output_data.gctx"):
    """
    Create a GCTX file with all genes and appropriate metadata structure.
    IMPORTANT: This function preserves the original gene order from the source GCTX file.
    
    Args:
        data_dict (dict): Dictionary containing processed data
        output_path (str): Path for the output GCTX file
    """
    print("Creating GCTX file with proper gene ordering...")
    
    # Extract necessary data from data_dict
    siginfo = data_dict['siginfo'].copy()
    geneinfo_ordered = data_dict['geneinfo_ordered']
    all_genes_list = data_dict['all_genes_list']
    
    # Keep track of signature IDs in the order they appear in siginfo
    sig_ids_ordered = siginfo['sig_id'].tolist()
    
    # Extract signature indices from gctx_slice_indices
    slice_1_idx = data_dict['gctx_indices_sigs_ids']['slice_1_idx']
    slice_2_idx = data_dict['gctx_indices_sigs_ids']['slices_2_idx']
    
    # Create matrices for each slice with original gene ordering
    print("Getting gene expression data in original gene order...")
    matrix_slice_1 = matrix[slice_1_idx.index, :]
    matrix_slice_2 = matrix[slice_2_idx.index, :]
    
    # Map signature IDs to their rows in the matrix slices
    sig_id_to_row_idx = {}
    for i, sig_id in enumerate(slice_1_idx.values):
        sig_id_to_row_idx[sig_id] = (0, i)  # (slice_number, row_index)
    
    for i, sig_id in enumerate(slice_2_idx.values):
        sig_id_to_row_idx[sig_id] = (1, i)  # (slice_number, row_index)
    
    # Build the matrix in the same order as siginfo
    print("Reordering expression data to match signature order...")
    reordered_matrix = np.zeros((len(sig_ids_ordered), matrix_slice_1.shape[1]), dtype=np.float32)
    
    for i, sig_id in enumerate(sig_ids_ordered):
        slice_num, row_idx = sig_id_to_row_idx[sig_id]
        if slice_num == 0:
            reordered_matrix[i, :] = matrix_slice_1[row_idx, :]
        else:
            reordered_matrix[i, :] = matrix_slice_2[row_idx, :]
    
    # Verify that the dimensions match expectations
    print(f"Matrix dimensions: {reordered_matrix.shape[0]} samples × {reordered_matrix.shape[1]} genes")
    print(f"Expected samples: {len(siginfo)}")
    print(f"Expected genes: {len(all_genes_list)}")
    
    # Make sure viability and dose are included
    siginfo['viability_clipped'] = np.clip(siginfo['viability'], 0, 1)
    siginfo['pert_dose_log2'] = np.log2(siginfo['pert_dose'])
    
    # Prepare column metadata (genes) - using original gene order
    col_metadata = pd.DataFrame({
        'gene_symbol': all_genes_list,
        'gene_id': geneinfo_ordered.index,
        'feature_space': geneinfo_ordered['feature_space']
    })
    
    # Create GCTX file
    print("Writing GCTX file...")
    with h5py.File(output_path, 'w') as f:
        # Set root attributes
        f.attrs['version'] = '1.0'
        f.attrs['description'] = 'Multimodal Drug Response Prediction Dataset'
        f.attrs['data_source'] = 'LINCS and CTRP integration'
        f.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.attrs['platform'] = 'LINCS L1000'
        
        # Create main groups
        group_0 = f.create_group('0')
        data_group = group_0.create_group('DATA')
        data_0_group = data_group.create_group('0')
        meta_group = group_0.create_group('META')
        row_meta_group = meta_group.create_group('ROW')
        col_meta_group = meta_group.create_group('COL')
        
        # Store expression matrix
        # We use reordered_matrix which maintains original gene order
        matrix_data = reordered_matrix.astype(np.float32)
        data_0_group.create_dataset(
            'matrix', 
            data=matrix_data,
            compression='gzip',
            chunks=(256, matrix_data.shape[1])
        )
        
        # Store row metadata
        for column in siginfo.columns:
            col_data = siginfo[column].values
            # Handle different data types
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                row_meta_group.create_dataset(column, data=col_data)
            else:
                # Convert strings to byte strings
                string_data = np.array([str(x).encode('utf-8') for x in col_data])
                max_length = max([len(x) for x in string_data])
                row_meta_group.create_dataset(
                    column, 
                    data=string_data,
                    dtype=f'S{max_length}'
                )
        
        # Store the row ids (sig_id)
        sig_ids_bytes = np.array([str(x).encode('utf-8') for x in siginfo['sig_id']])
        max_length = max([len(x) for x in sig_ids_bytes])
        row_meta_group.create_dataset('id', data=sig_ids_bytes, dtype=f'S{max_length}')
        
        # Store column metadata
        for column in col_metadata.columns:
            col_data = col_metadata[column].values
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                col_meta_group.create_dataset(column, data=col_data)
            else:
                # Convert strings to byte strings
                string_data = np.array([str(x).encode('utf-8') for x in col_data])
                max_length = max([len(x) for x in string_data])
                col_meta_group.create_dataset(
                    column, 
                    data=string_data,
                    dtype=f'S{max_length}'
                )
        
        # Store gene symbols as column ids
        gene_ids_bytes = np.array([str(x).encode('utf-8') for x in col_metadata['gene_id']])
        max_length = max([len(x) for x in gene_ids_bytes])
        col_meta_group.create_dataset('id', data=gene_ids_bytes, dtype=f'S{max_length}')
    
    # Validation check
    print("Validating GCTX file...")
    with h5py.File(output_path, 'r') as f:
        # Check dimensions
        stored_matrix = f['0']['DATA']['0']['matrix']
        print(f"Stored matrix dimensions: {stored_matrix.shape}")
        
        # Check first few gene IDs
        stored_gene_ids = [x.decode('utf-8') for x in f['0']['META']['COL']['id']]
        expected_gene_ids = [str(x) for x in geneinfo_ordered.index[:5]]
        print("First 5 gene IDs in stored file:", stored_gene_ids[:5])
        print("First 5 expected gene IDs:", expected_gene_ids)
        
        # Check a few signature IDs
        stored_sig_ids = [x.decode('utf-8') for x in f['0']['META']['ROW']['id']]
        expected_sig_ids = siginfo['sig_id'].astype(str).tolist()[:5]
        print("First 5 signature IDs in stored file:", stored_sig_ids[:5])
        print("First 5 expected signature IDs:", expected_sig_ids)
    
    print(f"GCTX file created successfully at: {output_path}")
    print(f"Dimensions: {matrix_data.shape[0]} samples × {matrix_data.shape[1]} genes")
    
    # Also save the key metadata for verification
    verification_data = {
        'gene_ids': [str(x) for x in geneinfo_ordered.index],
        'sig_ids': siginfo['sig_id'].tolist(),
        'dimensions': matrix_data.shape
    }
    with open(output_path + '.verification.json', 'w') as f:
        json.dump(verification_data, f)
    
    return True

# %%
# ======= Defining columns for matching ========

# Experiment id
experiment_id_list = curves_df.experiment_id.tolist()

# Experiment id -> Master ccl id
master_ccl_id_mapping = per_experiment.set_index('experiment_id')['master_ccl_id'].to_dict()

# Master ccl id -> ccl name
ccl_name_mapping = per_cell_line.set_index('master_ccl_id')['ccl_name'].to_dict()

# Master cpd id
master_cpd_id_list = curves_df.master_cpd_id.tolist()

# Master cpd id -> Compound name
cpd_name_mapping = per_compound.set_index('master_cpd_id')['cpd_name'].to_dict()

# Master cpd id -> Broad cpd id
broad_cpd_id_mapping = per_compound.set_index('master_cpd_id')['broad_cpd_id'].to_dict()

# (Series) Curves idx -> Smile
per_compound_smiles_curves_ordered = per_compound.set_index('master_cpd_id').loc[curves_df.master_cpd_id].cpd_smiles
per_compound_smiles_curves_ordered.index = curves_df.index

# %%
# ======= Mapping cl and cpd names to their positions in curves ========

# Master ccl id (in curves order)
master_ccl_id_list = [master_ccl_id_mapping[experiment_id] for experiment_id in experiment_id_list]

# Ccl name (in curves order)
ccl_name_list = [ccl_name_mapping[master_ccl_id] for master_ccl_id in master_ccl_id_list]

# Cpd name (in curves order)
cpd_name_list = [cpd_name_mapping[master_cpd_id] for master_cpd_id in master_cpd_id_list]

# Broad cpd id (in curves order)
broad_cpd_id_list = [broad_cpd_id_mapping[master_cpd_id] for master_cpd_id in master_cpd_id_list]

# %%
# ======= Getting siginfo cl and cpd name as lists ========

# Cell mfc name
cell_mfc_name_list = siginfo_df.cell_mfc_name.tolist()

# Pert mfc id (compound name 1)
pert_mfc_id_list = siginfo_df.pert_mfc_id.tolist()

# Cmap name (compound name 2)
cmap_name_list = siginfo_df.cmap_name.tolist()

# %%
# ======= Converting pairs to a list of tuples ========

# (List) (cell line name, compound name) (siginfo order)
siginfo_1 = list(zip(cell_mfc_name_list, pert_mfc_id_list))
siginfo_2 = list(zip(cell_mfc_name_list, cmap_name_list))

# (Series) (cell line name, compound name) (siginfo order)
siginfo_1_Series = pd.Series(siginfo_1)
siginfo_2_Series = pd.Series(siginfo_2)

# (List) (cell line name, compound name) (curve order)
ctrp_1 = list(zip(ccl_name_list, broad_cpd_id_list))
ctrp_2 = list(zip(ccl_name_list, cmap_name_list))

# (Series) (cell line name, compound name) (curve order)
ctrp_1_Series = pd.Series(ctrp_1)
ctrp_2_Series = pd.Series(ctrp_2)

# (set) (cell line name, compound name) (in siginfo and curves)
isct_1 = set(siginfo_1).intersection(set(ctrp_1))
isct_2 = set(siginfo_2).intersection(set(ctrp_2))

# (list) (cell line name, compound name) (in siginfo and curves)
isct_1_list = list(isct_1)
isct_2_list = list(isct_2)

# (Series) (cell line name, compound name) (in siginfo and curves) (with curve index)
ctrp_1_in_isct = ctrp_1_Series.loc[ctrp_1_Series.isin(isct_1)]
ctrp_2_in_isct = ctrp_2_Series.loc[ctrp_2_Series.isin(isct_2)]

# (Series) (cell line name, compound name) (in siginfo and curves) (with siginfo index)
siginfo_1_in_isct = siginfo_1_Series.loc[siginfo_1_Series.isin(isct_1)]
siginfo_2_in_isct = siginfo_2_Series.loc[siginfo_2_Series.isin(isct_2)]

# %%
# Get boolean array of indices if the siginfo series index is in the first matching
siginfo_2_idx_in_siginfo_1 = siginfo_2_in_isct.index.isin(siginfo_1_in_isct.index)

# Keep only the siginfo indices that are not in the series index of the first matching
siginfo_2_in_isct = siginfo_2_in_isct.loc[~(siginfo_2_idx_in_siginfo_1)]

# From the idices that are only in the second matching, acquire the unique (cell line name, compound name)
isct_2_not_covered_list = list(siginfo_2_in_isct.unique())

# Note that this is an IMPORTANT mapping that gets instantiated
ctrp_1_idx_mapping = get_ctrp_idx_mapping(siginfo_1_in_isct, ctrp_1_in_isct, isct_1_list)
ctrp_2_idx_mapping = get_ctrp_idx_mapping(siginfo_2_in_isct, ctrp_2_in_isct, isct_2_not_covered_list)

# Only the first curve index, as the drug will be identical, for easier handling
ctrp_1_idx_mapping_first = [i[0] for i in ctrp_1_idx_mapping.values()]
ctrp_2_idx_mapping_first = [i[0] for i in ctrp_2_idx_mapping.values()]

# Get the smiles using the curve indices (first matching)
per_compound_smiles_series_1 = pd.Series(per_compound_smiles_curves_ordered.loc[ctrp_1_idx_mapping_first])
per_compound_smiles_series_1.index = list(ctrp_1_idx_mapping.keys())

# Get the smiles using the curve indices (second matching)
per_compound_smiles_series_2 = pd.Series(per_compound_smiles_curves_ordered.loc[ctrp_2_idx_mapping_first])
per_compound_smiles_series_2.index = list(ctrp_2_idx_mapping.keys())

# Concatenate the smiles for both matchings
per_compound_smiles_concat = pd.concat([
            per_compound_smiles_series_1
            , per_compound_smiles_series_2]).tolist()

# Using the ctrp indices mapping, get the mapping from siginfo index to viability
viability_mapping_1 = get_viability_mapping(ctrp_1_idx_mapping)
viability_mapping_2 = get_viability_mapping(ctrp_2_idx_mapping)

# Take the mean of the viabilities when there were multiple curves for the siginfo index
viability_mapping_1_mean = {siginfo_i: np.array(viability_i).mean() for siginfo_i, viability_i in tqdm(viability_mapping_1.items())}
viability_mapping_2_mean = {siginfo_i: np.array(viability_i).mean() for siginfo_i, viability_i in tqdm(viability_mapping_2.items())}

# Concatenate the mean viability values from both matchings
viability_concat = list(viability_mapping_1_mean.values()) + list(viability_mapping_2_mean.values())

# Concatenate the two siginfo subsets using the matched siginfo indices
siginfo_concat = pd.concat([
    siginfo_df.loc[ctrp_1_idx_mapping.keys()]
    , siginfo_df.loc[ctrp_2_idx_mapping.keys()]])

# %%
# Acquiring the lowest concentration for each unique ctrp curve

# Drop the duplicates for optimised value fetching (the process took very long before)
ctrp_1_in_isct_drop_dup_reset_index = pd.DataFrame(ctrp_1_in_isct.drop_duplicates().reset_index(drop=False))
ctrp_2_in_isct_drop_dup_reset_index = pd.DataFrame(ctrp_2_in_isct.drop_duplicates().reset_index(drop=False))

# Use the curve indices to get the minimum value
ctrp_1_min = get_min(ctrp_1_in_isct_drop_dup_reset_index['index'].tolist())
ctrp_2_min = get_min(ctrp_2_in_isct_drop_dup_reset_index['index'].tolist())

# Add the minimum value to the dataframe with the tuples
ctrp_1_in_isct_drop_dup_reset_index['min'] = ctrp_1_min
ctrp_2_in_isct_drop_dup_reset_index['min'] = ctrp_2_min

# Set the tuples as index to order them according to the intersections, take the minimum value only
ctrp_1_min_list = list(ctrp_1_in_isct_drop_dup_reset_index.set_index(0).loc[list(siginfo_1_in_isct)]['min'])
ctrp_2_min_list = list(ctrp_2_in_isct_drop_dup_reset_index.set_index(0).loc[list(siginfo_2_in_isct)]['min'])

# Concatenate the lowest ctrp concentration
ctrp_min = ctrp_1_min_list + ctrp_2_min_list

# %%
# Add smiles, viability to the concatenated matched siginfo
siginfo_concat['smiles'] = per_compound_smiles_concat
siginfo_concat['viability'] = viability_concat
siginfo_concat['lowest_conc_log2'] = ctrp_min

# %%
# Get the siginfo slices (previously concatenated) from both matchings,
# We split these up in two again, to make the inference more computable
siginfo_slice_1 = siginfo_concat.loc[ctrp_1_idx_mapping.keys()]
siginfo_slice_2 = siginfo_concat.loc[ctrp_2_idx_mapping.keys()]

# Note that we deal here with the sig_ids with the indices from gctx
slice_1_idx = sig_ids.loc[sig_ids.isin(siginfo_slice_1.sig_id)]
slice_2_idx = sig_ids.loc[sig_ids.isin(siginfo_slice_2.sig_id)]

# Run the inference on the both slices by providing the gctx (numerical) indices
print("slice 1")
acts_slice_1 = run_inference_on_slice(slice_1_idx)
print("slice 2")
acts_slice_2 = run_inference_on_slice(slice_2_idx)

# Slice indices for gctx matrix, for later reference
gctx_slice_indices = {'slice_1_idx': slice_1_idx, 'slices_2_idx': slice_2_idx}

# %%
# Getting all genes from the matrix (first matching)
print("genes slice 1..")
matrix_slice_1 = matrix[slice_1_idx.index,:]
matrix_slice_1_lm = matrix_slice_1[:, landmark_idxs]

# Getting all genes from the matrix (second matching)
print("genes slice 2..")
matrix_slice_2 = matrix[slice_2_idx.index,:]
matrix_slice_2_lm = matrix_slice_2[:, landmark_idxs]

# Concatenate the matrix slices for all the genes into a single numpy array
all_genes_concat = np.concatenate((matrix_slice_1, matrix_slice_2))

# Convert the matrix slices for the landmark genes
matrix_slice_1_lm_df = pd.DataFrame(matrix_slice_1_lm, index=slice_1_idx.tolist())
matrix_slice_2_lm_df = pd.DataFrame(matrix_slice_2_lm, index=slice_2_idx.tolist())

# Concatenate the matrix slices with the landmark genes into a single dataframe
matrix_lm_concat = pd.concat([matrix_slice_1_lm_df, matrix_slice_2_lm_df])

# %%
# Concatenate the inferred activity dataframes
acts_concat = pd.concat([acts_slice_1[0], acts_slice_2[0]])

# Create single-column dataframe with the signature ids as indices and their position in the concatenated landmark matrix as the values
matrix_lm_enumerated_index = pd.Series(matrix_lm_concat.index).reset_index().set_index(0)

# Get the dataframe with values as indices order of the concatenated activities
matrix_acts_concat_index_num = matrix_lm_enumerated_index.loc[acts_concat.index]['index'].tolist()

# Apply the reordering to the all genes numpy array
all_genes_acts_order = all_genes_concat[matrix_acts_concat_index_num, :]

# Place siginfo in the order of acts_concat
siginfo_acts_order = siginfo_concat.set_index('sig_id', drop=False).loc[acts_concat.index]

# Place matrix lm in the same order as acts_concat
matrix_lm_acts_order = matrix_lm_concat.loc[acts_concat.index]

# Create a list with the clipped viability values
viability_acts_order_clipped = siginfo_acts_order.viability.clip(0,1).tolist()

# Create a list with the doses (log2-transformed)
pert_dose_log2_acts_order = np.log2(siginfo_acts_order.pert_dose).tolist()

# %%
# Compile the different data into a dictionary
data_dict_74709 = {
    'acts': acts_concat,
    'siginfo': siginfo_acts_order,
    'gctx_indices_sigs_ids': gctx_slice_indices,
    'all_genes': all_genes_acts_order,  # This contains all genes (not just landmark)
    'lm': matrix_lm_acts_order,
    'viability_clipped': viability_acts_order_clipped,
    'pert_dose_log2': pert_dose_log2_acts_order,
    'geneinfo_ordered': geneinfo_ordered,
    'lm_genes_list': landmark_gene_symbol,
    'all_genes_list': all_gene_symbol
}

# Save as pickle first (for comparison with original)
with open('pk/data_dict_74709.pk', 'wb') as handle:
    pk.dump(data_dict_74709, handle)
    print("Pickle file saved for comparison.")

# Create GCTX file with all genes and proper metadata
create_gctx_file(data_dict_74709, output_path="data/processed/LINCS_CTRP_QC2.gctx")

print("Processing complete. GCTX file created with correct gene ordering.")

# Optional: Verification step to compare new GCTX with original pickle
try:
    # Load original pickle if it exists
    original_pickle_path = 'pk/data_dict_74709_original.pk'
    import os
    if os.path.exists(original_pickle_path):
        with open(original_pickle_path, 'rb') as handle:
            original_data = pk.load(handle)
        
        # Compare gene ordering
        same_genes = np.array_equal(original_data['geneinfo_ordered'].index, data_dict_74709['geneinfo_ordered'].index)
        print(f"Gene IDs match with original pickle: {same_genes}")
        
        # Compare matrix dimensions
        original_shape = original_data['all_genes'].shape
        new_shape = data_dict_74709['all_genes'].shape
        print(f"Original matrix shape: {original_shape}")
        print(f"New matrix shape: {new_shape}")
        
        # Check a few values
        if original_shape == new_shape:
            # Check first sample, first 5 genes
            print("First 5 genes of first sample (original):", original_data['all_genes'][0, :5])
            print("First 5 genes of first sample (new):", data_dict_74709['all_genes'][0, :5])
except Exception as e:
    print(f"Could not perform verification against original pickle: {e}")