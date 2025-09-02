import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from scripts.train import create_data_loaders, initialize_model
from utils.config_parser import ConfigParser
from __init__ import path_to_config, path_to_project
from utils.custom_logging import get_logger

logger = get_logger(__name__)


def plot_results(config_dict=None):
    if config_dict is None:
        config = ConfigParser().parse(path_to_config())
    else:
        config = config_dict
        
    train_config = config['train']
    paths_config = config['paths']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics_path = os.path.join(paths_config['metrics_path'], f"{train_config['experiment_type']}_metrics.pt")
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found: {metrics_path}")
        return
        
    metrics = torch.load(metrics_path)
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    val_accuracies = metrics['val_accuracies']

    os.makedirs(paths_config['plots_path'], exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(paths_config['plots_path'], 'training_metrics.png'))
    plt.close()
    
    model = initialize_model(config)
    model_path = os.path.join(paths_config['weights_path'], f"{train_config['experiment_type']}_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _, _, test_loader = create_data_loaders(config)

    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            tabulars = batch['tabulars'].to(device)
            targets = batch['targets'].to(device)

            if train_config['experiment_type'] == 'image':
                outputs = model(images)
            elif train_config['experiment_type'] == 'tabular':
                outputs = model(tabulars)
            else:
                outputs = model(images, tabulars)

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(paths_config['plots_path'], 'confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(paths_config['plots_path'], 'roc_curve.png'))
    plt.close()

    report = classification_report(all_targets, all_predictions, target_names=['Class 0', 'Class 1'])
    with open(os.path.join(paths_config['plots_path'], 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    metrics_text = f"""
    Experiment Type: {train_config['experiment_type']}
    Final Validation Accuracy: {val_accuracies[-1]:.2f}%
    Best Validation Accuracy: {max(val_accuracies):.2f}%
    ROC AUC: {roc_auc:.4f}
    
    Classification Report:
    {report}
    """
    
    with open(os.path.join(paths_config['plots_path'], 'experiment_metrics.txt'), 'w') as f:
        f.write(metrics_text)

    logger.info(f"Plots and metrics saved to {paths_config['plots_path']}")
    logger.info(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    logger.info(f"ROC AUC: {roc_auc:.4f}")


if __name__ == '__main__':
    plot_results()