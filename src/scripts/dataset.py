import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, data_path, image_folder, metadata_file, target_column=None, target_percentile=0.8, transform=None):
        self.data_path = data_path
        self.image_folder = image_folder
        self.metadata = pd.read_csv(os.path.join(data_path, metadata_file), engine='python', on_bad_lines='warn')
        self.transform = transform

        if target_column and target_column in self.metadata.columns:
            if self.metadata[target_column].dtype in ['int64', 'float64']:
                if set(self.metadata[target_column].unique()).issubset({0, 1, 0.0, 1.0}):
                    self.metadata['target'] = self.metadata[target_column].astype(int)
                else:
                    threshold = self.metadata[target_column].quantile(target_percentile)
                    self.metadata['target'] = (self.metadata[target_column] >= threshold).astype(int)
            else:
                unique_vals = self.metadata[target_column].unique()
                self.metadata['target'] = (self.metadata[target_column] != unique_vals[0]).astype(int)
        else:
            import warnings
            warnings.warn(f"No target column '{target_column}' found and no 'price' column available. "
                         f"Generating synthetic CTR target based on available product attributes.", 
                         UserWarning)
            self.metadata['target'] = self._generate_ctr_target()

        categorical_cols = []
        for col in self.metadata.columns:
            if self.metadata[col].dtype == 'object' and col not in ['id', 'target', 'productDisplayName']:
                categorical_cols.append(col)
        
        if categorical_cols:
            self.metadata = pd.get_dummies(self.metadata, columns=categorical_cols)
        
        self.feature_columns = [col for col in self.metadata.columns if col not in ['id', 'target', 'productDisplayName']]

    def _generate_ctr_target(self):
        import random
        random.seed(42)
        
        base_success_rate = 0.2
        target = []
        
        for idx, row in self.metadata.iterrows():
            success_prob = base_success_rate
            
            if 'gender' in self.metadata.columns:
                if row['gender'] == 'Women':
                    success_prob *= 1.2
                elif row['gender'] == 'Men':
                    success_prob *= 1.1
            
            if 'masterCategory' in self.metadata.columns:
                if 'Apparel' in str(row['masterCategory']):
                    success_prob *= 1.15
            
            success_prob += random.uniform(-0.1, 0.1)
            success_prob = max(0.05, min(0.95, success_prob))  # Clamp between 5% and 95%
            
            target.append(1 if random.random() < success_prob else 0)
        
        return target

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.data_path, self.image_folder, f"{img_id}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                processed = self.transform(images=image, return_tensors="pt")
                image_tensor = processed['pixel_values'].squeeze(0)
            else:
                import torchvision.transforms as transforms
                basic_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                image_tensor = basic_transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image_tensor = torch.zeros((3, 224, 224))

        tabular_data = row[self.feature_columns].values.astype(float)
        tabular_data = torch.tensor(tabular_data, dtype=torch.float32)

        target = row['target']
        return {
            'image': image_tensor,
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