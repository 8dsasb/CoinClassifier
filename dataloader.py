# dataloader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np

class CoinImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Build full image path from folder + image name
        img_folder = row['URL']
        img_name = str(row['image name']).strip()
        img_path = img_folder +'/'+ img_name
        label = row['encoded_class']

        image = Image.open(img_path).convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0


        return image, label


def load_coin_dataset(csv_url, batch_size=32):
    """
    Loads the coin dataset from CSV and prepares PyTorch datasets and dataloaders.

    Args:
        csv_url (str): URL or local path to the CSV file
        batch_size (int): batch size for the loaders

    Returns:
        train_df, val_df, test_df (DataFrames)
        train_loader, val_loader, test_loader (PyTorch DataLoaders)
        label_encoder (LabelEncoder to inverse labels)
    """
    df = pd.read_csv(csv_url)
    df.columns = df.columns.str.strip()  # Clean column names

    # Drop any rows with missing or malformed data
    df.dropna(subset=['URL', 'image name', 'label'], inplace=True)

    # Encode the class labels
    label_encoder = LabelEncoder()
    df['encoded_class'] = label_encoder.fit_transform(df['label'])

    # Split into train (60%), val (10%), test (30%)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['encoded_class'])
    val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42, stratify=temp_df['encoded_class'])
    print(train_df)
    # Build datasets
    train_dataset = CoinImageDataset(train_df)
    val_dataset = CoinImageDataset(val_df)
    test_dataset = CoinImageDataset(test_df)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader
