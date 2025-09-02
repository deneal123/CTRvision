
# CTRvision

AI model that helps marketers choose better product photos by predicting their performance. Increases CTR and conversions using computer vision and machine learning.

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
