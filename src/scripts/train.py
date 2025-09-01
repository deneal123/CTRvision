import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from src.model.classification_head import ImageClassifier, TabularClassifier, CombinedClassifier
from src.model.focal_loss import FocalLoss
from src.scripts.dataset import CustomDataset, collate_fn
from src.utils.config_parser import ConfigParser
from src import path_to_config, path_to_project
from src.utils.custom_logging import get_logger

logger = get_logger(__name__)

def create_data_loaders(config):
    """Create train, validation, and test data loaders."""
    data_config = config['data']
    train_config = config['train']
    
    transform = AutoImageProcessor.from_pretrained(config['model']['image_model_name']).preprocess
    dataset = CustomDataset(
        data_path=os.path.join(path_to_project(), data_config['data_path']),
        image_folder=data_config['image_folder'],
        metadata_file=data_config['metadata_file'],
        target_column=data_config.get('target_column', None),
        target_percentile=data_config.get('target_percentile', 0.8),
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(train_config['seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=train_config['num_workers'], pin_memory=train_config['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=train_config['num_workers'], pin_memory=train_config['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=train_config['num_workers'], pin_memory=train_config['pin_memory'])
    
    return train_loader, val_loader, test_loader

def initialize_model(config):
    """Initialize model based on experiment type."""
    train_config = config['train']
    model_config = config['model']
    data_config = config['data']
    
    # Create temporary dataset to get input dimensions
    transform = AutoImageProcessor.from_pretrained(model_config['image_model_name']).preprocess
    dataset = CustomDataset(
        data_path=os.path.join(path_to_project(), data_config['data_path']),
        image_folder=data_config['image_folder'],
        metadata_file=data_config['metadata_file'],
        target_column=data_config.get('target_column', None),
        target_percentile=data_config.get('target_percentile', 0.8),
        transform=transform
    )
    
    experiment_type = train_config['experiment_type']
    if experiment_type == 'image':
        model = ImageClassifier(model_config['image_model_name'], num_classes=2)
    elif experiment_type == 'tabular':
        sample = dataset[0]
        tabular_input_size = sample['tabular'].shape[0]
        model = TabularClassifier(tabular_input_size, model_config['tabular_hidden_sizes'], num_classes=2)
    elif experiment_type == 'combined':
        sample = dataset[0]
        tabular_input_size = sample['tabular'].shape[0]
        model = CombinedClassifier(
            model_config['image_model_name'],
            tabular_input_size,
            model_config['combined_hidden_sizes'],
            num_classes=2
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return model

def train():
    config = ConfigParser().parse(path_to_config())
    data_config = config['data']
    train_config = config['train']
    model_config = config['model']
    paths_config = config['paths']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use helper functions for consistency
    train_loader, val_loader, test_loader = create_data_loaders(config)
    model = initialize_model(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    
    # Use FocalLoss instead of CrossEntropyLoss for better handling of imbalanced classes
    focal_loss_gamma = train_config.get('focal_loss_gamma', 2.0)
    criterion = FocalLoss(gamma=focal_loss_gamma)

    experiment_type = train_config['experiment_type']
    
    for epoch in range(train_config['num_epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch['images'].to(device)
            tabulars = batch['tabulars'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()

            if experiment_type == 'image':
                outputs = model(images)
            elif experiment_type == 'tabular':
                outputs = model(tabulars)
            else:
                outputs = model(images, tabulars)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                tabulars = batch['tabulars'].to(device)
                targets = batch['targets'].to(device)

                if experiment_type == 'image':
                    outputs = model(images)
                elif experiment_type == 'tabular':
                    outputs = model(tabulars)
                else:
                    outputs = model(images, tabulars)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {accuracy:.2f}%')

    os.makedirs(paths_config['weights_path'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(paths_config['weights_path'], f'{experiment_type}_model.pth'))

if __name__ == '__main__':
    train()