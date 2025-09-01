import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, data_path, image_folder, metadata_file, target_column=None, target_percentile=0.8, transform=None):
        self.data_path = data_path
        self.image_folder = image_folder
        self.metadata = pd.read_csv(os.path.join(data_path, metadata_file))
        self.transform = transform

        # More universal target column handling
        if target_column and target_column in self.metadata.columns:
            # If target column specified and exists, use it directly
            if self.metadata[target_column].dtype in ['int64', 'float64']:
                # Numeric target - assume it's already binary or needs thresholding
                if set(self.metadata[target_column].unique()).issubset({0, 1, 0.0, 1.0}):
                    self.metadata['target'] = self.metadata[target_column].astype(int)
                else:
                    # Apply percentile-based thresholding for numeric targets
                    threshold = self.metadata[target_column].quantile(target_percentile)
                    self.metadata['target'] = (self.metadata[target_column] >= threshold).astype(int)
            else:
                # Categorical target - encode as binary (0 for first unique value, 1 for others)
                unique_vals = self.metadata[target_column].unique()
                self.metadata['target'] = (self.metadata[target_column] != unique_vals[0]).astype(int)
        elif 'price' in self.metadata.columns:
            # Default behavior: use price-based target
            threshold = self.metadata['price'].quantile(target_percentile)
            self.metadata['target'] = (self.metadata['price'] >= threshold).astype(int)
        else:
            # Generate synthetic CTR target if no suitable column found
            print(f"No target column '{target_column}' found. Generating synthetic CTR target...")
            # Create a proxy CTR metric based on product attributes
            self.metadata['target'] = self._generate_ctr_target()

        # Handle categorical columns more universally
        categorical_cols = []
        for col in self.metadata.columns:
            if self.metadata[col].dtype == 'object' and col not in ['id', 'target', 'productDisplayName']:
                categorical_cols.append(col)
        
        if categorical_cols:
            self.metadata = pd.get_dummies(self.metadata, columns=categorical_cols)
        
        self.feature_columns = [col for col in self.metadata.columns if col not in ['id', 'target', 'productDisplayName']]

    def _generate_ctr_target(self):
        """Generate a synthetic CTR target based on product attributes."""
        # Set random seed for reproducible results
        import random
        random.seed(42)
        
        # Base CTR success rate
        base_success_rate = 0.2
        target = []
        
        for idx, row in self.metadata.iterrows():
            success_prob = base_success_rate
            
            # Adjust probability based on available attributes
            if 'price' in self.metadata.columns:
                # Higher prices might have lower CTR
                price_factor = max(0.1, 1 - (row['price'] / self.metadata['price'].max()) * 0.3)
                success_prob *= price_factor
            
            if 'gender' in self.metadata.columns:
                # Different CTR for different genders
                if row['gender'] == 'Women':
                    success_prob *= 1.2
                elif row['gender'] == 'Men':
                    success_prob *= 1.1
            
            if 'masterCategory' in self.metadata.columns:
                # Different CTR for different categories
                if 'Apparel' in str(row['masterCategory']):
                    success_prob *= 1.15
            
            # Add some randomness
            success_prob += random.uniform(-0.1, 0.1)
            success_prob = max(0.05, min(0.95, success_prob))  # Clamp between 5% and 95%
            
            # Generate binary target
            target.append(1 if random.random() < success_prob else 0)
        
        return target

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