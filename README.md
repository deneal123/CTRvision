
# CTRvision

AI model that helps marketers choose better product photos by predicting their performance. Increases CTR and conversions using computer vision and machine learning.

## 🚀 Recent Improvements

This project has been enhanced with several key improvements:

- **🎯 FocalLoss Integration**: Replaced standard CrossEntropyLoss with FocalLoss for better handling of imbalanced datasets
- **🔧 Universal Dataset**: Enhanced dataset class to handle various target columns and generate synthetic CTR targets
- **🔗 Synergistic Scripts**: Added helper functions to link all scripts together seamlessly
- **📊 Three Experiment Types**: Support for image-only, tabular-only, and combined multi-modal approaches
- **🎛️ Configurable Pipeline**: Enhanced configuration with focal loss parameters and flexible target column selection

## 📋 Quick Start

### 1. Installation

Clone the repository:
```bash
git clone https://github.com/deneal123/CTRvision.git
cd CTRvision
```

Install dependencies:
```bash
bash install.sh
```

*Warning: If you still can't run scripts, follow these steps:*
```bash
rm -rf ./.venv
pyenv init # follows instructions
pyenv shell 3.10.6
uv cache clean
uv python pin 3.10.6
uv sync
```
*Make sure you are using Python 3.10.6.*

### 2. Run Complete Pipeline

```bash
# Run the full pipeline (download -> train -> evaluate)
python src/scripts/main.py --step all

# Or run individual steps
python src/scripts/main.py --step download
python src/scripts/main.py --step train
python src/scripts/main.py --step plot
```

### 3. Validate Installation

```bash
python validate_improvements.py
```

## 🧪 Experiments

The project supports three different experiment configurations. See [experiments_guide.md](experiments_guide.md) for detailed instructions:

1. **Image-Only Model**: Uses only product images with DINOv2 features
2. **Tabular-Only Model**: Uses only product metadata (price, category, etc.)
3. **Combined Multi-Modal Model**: Combines both image and tabular features

## 🔧 Configuration

Key configuration options in `src/config.yaml`:

```yaml
data:
  target_column: null  # Specify target column or null for auto-detection
  target_percentile: 0.8  # Threshold for binary classification

train:
  experiment_type: "image"  # "image", "tabular", or "combined"
  focal_loss_gamma: 2.0  # Focal loss gamma parameter
  learning_rate: 0.001
  num_epochs: 10

model:
  num_classes: 2  # Binary classification
  unfreeze_layers: 0  # Fine-tuning parameter
```

## 🏃‍♂️ Local Development

Start Jupyter notebook:
```bash
bash notebook.sh
```

## 🐳 Docker

Build and run with Docker:
```bash
bash build.sh
bash run.sh
```

## 📁 Project Structure

```
CTRvision/
├── src/
│   ├── model/
│   │   ├── classification_head.py  # Model architectures
│   │   └── focal_loss.py          # FocalLoss implementation
│   ├── scripts/
│   │   ├── main.py                # Main pipeline orchestrator
│   │   ├── train.py               # Training script with FocalLoss
│   │   ├── dataset.py             # Universal dataset class
│   │   ├── plot.py                # Results visualization
│   │   └── download.py            # Data downloading
│   ├── utils/                     # Utility functions
│   └── config.yaml                # Configuration file
├── experiments_guide.md           # Experiment instructions
└── validate_improvements.py       # Validation script
```

## 🎯 Key Features

- **FocalLoss**: Better handling of imbalanced CTR data
- **Universal Dataset**: Automatic target generation for any product dataset
- **Multi-Modal**: Combines vision and tabular data
- **Configurable**: Easy experiment configuration via YAML
- **Production Ready**: Modular, tested, and documented code
