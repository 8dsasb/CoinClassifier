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
        folder_path = Path(row['URL'])  # This is a directory, not a file
        label = row['encoded_class']

        if not folder_path.is_dir():
            raise ValueError(f"Expected a directory but got: {folder_path}")

        # Search for image files inside the directory
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png"))

        if not image_files:
            raise FileNotFoundError(f"No image files found in: {folder_path}")

        # Use the first valid image file
        img_path = image_files[0]

        # Ensure img_path is a file (not a misnamed folder)
        if img_path.is_dir():
            raise ValueError(f"Found a directory instead of an image file: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = T.ToTensor()(image)

        return image, label
def load_coin_dataset(csv_url, batch_size=32):
    """
    Loads the coin classification dataset from a CSV file.

    Parameters:
        csv_url (str): Path or URL to the CSV file.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        train_df, val_df, test_df : pd.DataFrame
        train_loader, val_loader, test_loader : torch.utils.data.DataLoader
        label_encoder : sklearn.preprocessing.LabelEncoder
    """
    # Load CSV
    df = pd.read_csv(csv_url)

    # Encode class labels
    label_encoder = LabelEncoder()
    df['encoded_class'] = label_encoder.fit_transform(df['label'])

    # Split into train (60%), val (10%), test (30%)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['encoded_class'])
    val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42, stratify=temp_df['encoded_class'])

    # Create datasets
    train_dataset = CoinImageDataset(train_df)
    val_dataset = CoinImageDataset(val_df)
    test_dataset = CoinImageDataset(test_df)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader, label_encoder
