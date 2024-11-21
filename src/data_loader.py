import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TFViabilityDataset(Dataset):
    def __init__(self, x_file_path, y_file_path, transform=None, standardize=True):
        """
        Args:
            x_file_path (str): Path to the X .tsv file.
            y_file_path (str): Path to the Y .tsv file.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.X_data = pd.read_csv(x_file_path, sep="\t")
        self.y_data = pd.read_csv(y_file_path, sep="\t")

        # Use the first column as sample IDs
        self.sample_ids = self.X_data.iloc[:, 0].values

        # Drop the first column (sample IDs) from the feature data
        self.X_data = self.X_data.drop(columns=[self.X_data.columns[0]])

        # Extract target variable and additional feature
        self.y = self.y_data["viability"].values.astype(np.float32)
        self.X_data["pert_dose"] = self.y_data["pert_dose"].values.astype(np.float32)

        # Drop rows where the target variable 'viability' is NaN
        valid_indices = ~np.isnan(self.y)
        self.X_data = self.X_data.loc[valid_indices]
        self.y = self.y[valid_indices]
        self.sample_ids = self.sample_ids[valid_indices]

        # Handle missing values in features
        self.X_data.fillna(0, inplace=True)

        if standardize:
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.X_data).astype(np.float32)

        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        features = self.features[idx]
        target = self.y[idx]

        if self.transform:
            features = self.transform(features)

        return {"id": sample_id, "features": features, "target": target}
