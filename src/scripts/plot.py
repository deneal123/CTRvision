import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from src.scripts.train import create_data_loaders, initialize_model
from src.utils.config_parser import ConfigParser
from src import path_to_config, path_to_project
from src.utils.custom_logging import get_logger

logger = get_logger(__name__)


def plot_results():
    config = ConfigParser().parse(path_to_config())
    train_config = config['Train']
    paths_config = config['Paths']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = create_data_loaders(config)

    model = initialize_model(config)
    model_path = os.path.join(paths_config['weights_path'], f"{train_config['experiment_type']}_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []

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

            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(paths_config['plots_path'], 'confusion_matrix.png'))
    plt.close()

    report = classification_report(all_targets, all_predictions, target_names=['Class 0', 'Class 1'])
    with open(os.path.join(paths_config['plots_path'], 'classification_report.txt'), 'w') as f:
        f.write(report)

    logger.info("Plots and report saved.")

if __name__ == '__main__':
    plot_results()