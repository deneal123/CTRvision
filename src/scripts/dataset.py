import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, data_path, image_folder, metadata_file, target_percentile=0.8, transform=None):
        self.data_path = data_path
        self.image_folder = image_folder
        self.metadata = pd.read_csv(os.path.join(data_path, metadata_file))
        self.transform = transform

        if 'price' in self.metadata.columns:
            threshold = self.metadata['price'].quantile(target_percentile)
            self.metadata['target'] = (self.metadata['price'] >= threshold).astype(int)
        else:
            self.metadata['target'] = 0
            self.metadata.loc[self.metadata.sample(frac=0.2, random_state=42).index, 'target'] = 1

        self.categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
        self.metadata = pd.get_dummies(self.metadata, columns=self.categorical_columns)
        self.feature_columns = [col for col in self.metadata.columns if col not in ['id', 'target', 'productDisplayName']]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.data_path, self.image_folder, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tabular_data = row[self.feature_columns].values.astype(float)
        tabular_data = torch.tensor(tabular_data, dtype=torch.float32)

        target = row['target']
        return {
            'image': image,
            'tabular': tabular_data,
            'target': target
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    tabulars = [item['tabular'] for item in batch]
    targets = [item['target'] for item in batch]

    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)

    tabulars = torch.stack(tabulars)
    targets = torch.tensor(targets, dtype=torch.long)
    return {
        'images': images,
        'tabulars': tabulars,
        'targets': targets
    }