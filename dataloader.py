# dataloader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
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
        img_folder = row['URL']  # Folder containing the image
        label = row['encoded_class']

        # Find a .jpg file in the folder (assumes 1 image per folder or consistent naming)
        from pathlib import Path
        img_path = next(Path(img_folder).glob("*.jpg"))  # Add png/jpeg as needed

        image = Image.open(img_path).convert("RGB")
        image = T.ToTensor()(image)
        return image, label

def load_coin_dataset(csv_url, batch_size=32):
    df = pd.read_csv(csv_url)

    label_encoder = LabelEncoder()
    df['encoded_class'] = label_encoder.fit_transform(df['label'])

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['encoded_class'])
    val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42, stratify=temp_df['encoded_class'])

    train_dataset = CoinImageDataset(train_df)
    val_dataset = CoinImageDataset(val_df)
    test_dataset = CoinImageDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader, label_encoder
