# CTRvision Experiment Guide

This document provides instructions for running three different experimental configurations with the CTRvision project. Each experiment tests a different model architecture approach for predicting product CTR performance.

## Prerequisites

1. Ensure you have the dataset downloaded:
   ```bash
   python src/scripts/main.py --step download
   ```

2. Make sure all dependencies are installed and the environment is set up.

## Experiment 1: Image-Only Model

This experiment uses only product images to predict CTR performance.

### Configuration Changes

Edit `src/config.yaml` and modify the following sections:

```yaml
train:
  experiment_type: "image"
  batch_size: 32
  num_workers: 4
  learning_rate: 0.001
  num_epochs: 15
  focal_loss_gamma: 2.0

model:
  image_model_name: "facebook/dinov2-base"
```

### Run Command
```bash
python src/scripts/main.py --step train
python src/scripts/main.py --step plot
```

### Expected Results
- Model will use only image features from the pre-trained DINOv2 model
- Output files: `weights/image_model.pth`
- Evaluation plots in `plots/` directory

---

## Experiment 2: Tabular-Only Model

This experiment uses only product metadata (categorical and numerical features) to predict CTR performance.

### Configuration Changes

Edit `src/config.yaml` and modify the following sections:

```yaml
train:
  experiment_type: "tabular"
  batch_size: 64
  num_workers: 2
  learning_rate: 0.002
  num_epochs: 20
  focal_loss_gamma: 1.5

model:
  tabular_hidden_sizes: [512, 256, 128]
```

### Run Command
```bash
python src/scripts/main.py --step train
python src/scripts/main.py --step plot
```

### Expected Results
- Model will use only tabular features (price, gender, category, etc.)
- Output files: `weights/tabular_model.pth`
- Typically faster training due to smaller model size

---

## Experiment 3: Combined Multi-Modal Model

This experiment combines both image and tabular features for prediction.

### Configuration Changes

Edit `src/config.yaml` and modify the following sections:

```yaml
train:
  experiment_type: "combined"
  batch_size: 24  # Smaller batch size due to larger model
  num_workers: 4
  learning_rate: 0.0008
  num_epochs: 12
  focal_loss_gamma: 2.5

model:
  image_model_name: "facebook/dinov2-base"
  tabular_hidden_sizes: [512, 256]
  combined_hidden_sizes: [512, 256]
```

### Run Command
```bash
python src/scripts/main.py --step train
python src/scripts/main.py --step plot
```

### Expected Results
- Model will use both image and tabular features
- Output files: `weights/combined_model.pth`
- Usually achieves the best performance but requires more computational resources

---

## Target Column Experiments

You can also experiment with different target columns by modifying the data configuration:

### Using Price-Based Target (default)
```yaml
data:
  target_column: null  # Will auto-detect and use 'price'
  target_percentile: 0.8  # Top 20% of prices = successful CTR
```

### Using Custom Target Column
```yaml
data:
  target_column: "rating"  # If your dataset has a rating column
  target_percentile: 0.7  # Top 30% of ratings = successful CTR
```

### Using Synthetic CTR Target
```yaml
data:
  target_column: "synthetic_ctr"  # Non-existent column will trigger synthetic generation
  target_percentile: 0.8
```

---

## Performance Comparison

After running all experiments, compare the results:

1. **Accuracy**: Check the validation accuracy in the training logs
2. **Loss Curves**: Compare final validation losses
3. **Confusion Matrices**: Review `plots/confusion_matrix.png` for each experiment
4. **Classification Reports**: Check `plots/classification_report.txt` for detailed metrics

### Expected Performance Hierarchy
1. **Combined Model**: Best overall performance (highest accuracy)
2. **Image Model**: Good performance, especially for visual products
3. **Tabular Model**: Baseline performance, fastest to train

## Focal Loss Benefits

All experiments use Focal Loss instead of standard Cross-Entropy Loss, which provides:
- Better handling of class imbalance
- Focus on hard-to-classify examples
- Improved performance on minority class (successful CTR)

Adjust `focal_loss_gamma` parameter:
- Higher values (2.0-3.0): More focus on hard examples
- Lower values (1.0-1.5): More similar to standard cross-entropy

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch_size
2. **Slow Training**: Reduce num_workers or use smaller model
3. **Poor Performance**: Try different learning_rate or increase num_epochs

### Performance Tips
1. Use GPU if available for faster training
2. Adjust batch_size based on available memory
3. Monitor validation loss to avoid overfitting
4. Try different focal_loss_gamma values for imbalanced datasets