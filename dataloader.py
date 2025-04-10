# coin_dataset_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt


class CoinImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['URL']
        label = row['encoded_class']

        image = Image.open(img_path).convert("RGB")
        image = T.ToTensor()(image)
        return image, label


def load_coin_dataset(csv_url, batch_size=32):
    df = pd.read_csv(csv_url)

    # Encode class labels
    label_encoder = LabelEncoder()
    df['encoded_class'] = label_encoder.fit_transform(df['label'])

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['encoded_class'])
    val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42, stratify=temp_df['encoded_class'])

    # Datasets and Dataloaders
    train_dataset = CoinImageDataset(train_df)
    val_dataset = CoinImageDataset(val_df)
    test_dataset = CoinImageDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader, label_encoder


import os
from pathlib import Path

def show_images_from_df(df, n=10):
    subset = df.iloc[:n]

    plt.figure(figsize=(15, 5))
    count = 0

    for row in subset.itertuples():
        folder_path = getattr(row, 'URL')
        label = getattr(row, 'label')

        try:
            # Try to find the first image inside the folder
            image_files = list(Path(folder_path).glob("*.jpg"))  # You can add png, jpeg as needed

            if not image_files:
                print(f"No image found in {folder_path}")
                continue

            img = Image.open(image_files[0]).convert("RGB")

            plt.subplot(2, 5, count + 1)
            plt.imshow(img)
            plt.title(label, fontsize=9)
            plt.axis('off')
            count += 1

        except Exception as e:
            print(f"Error opening image from {folder_path}: {e}")

    plt.tight_layout()
    plt.show()
