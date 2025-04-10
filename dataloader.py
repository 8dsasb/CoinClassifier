# dataloader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch

class CoinImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Ensure image name is valid
        if pd.isna(row['image name']):
            raise ValueError(f"Missing image name for row: {row}")

        # Combine URL and image name
        img_path = Path(row['URL']) / row['image name']
        label = row['encoded_class']

        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found or not a file: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = T.ToTensor()(image)

        return image, label


def load_coin_dataset(csv_url, batch_size=32):
    """
    Loads the coin dataset from a CSV file and prepares DataLoaders.

    Parameters:
    - csv_url (str): URL or path to the CSV file
    - batch_size (int): Batch size for DataLoaders

    Returns:
    - train_df, val_df, test_df (DataFrames)
    - train_loader, val_loader, test_loader (DataLoaders)
    - label_encoder (for inverse transform if needed)
    """

    df = pd.read_csv(csv_url)
    df.columns = df.columns.str.strip()  # Clean column names

    # Drop rows with missing image names
    df.dropna(subset=['image name'], inplace=True)

    # Optional: remove rows with broken paths
    df['full_path'] = df.apply(lambda r: Path(r['URL']) / r['image name'], axis=1)
    df = df[df['full_path'].apply(lambda p: p.is_file())]
    df.drop(columns=['full_path'], inplace=True)

    # Encode class labels
    label_encoder = LabelEncoder()
    df['encoded_class'] = label_encoder.fit_transform(df['label'])

    # Split: 60% train, 10% val, 30% test
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['encoded_class'])
    val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42, stratify=temp_df['encoded_class'])

    # Datasets and loaders
    train_dataset = CoinImageDataset(train_df)
    val_dataset = CoinImageDataset(val_df)
    test_dataset = CoinImageDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader, label_encoder
