# Currency Banknote Classification - CNN Project

## Project Overview
This project implements a Convolutional Neural Network (CNN) for automated classification of currency banknotes. The model can identify different banknote denominations based on visual features like color, size, and security patterns.

## Dataset
- **Source**: Kaggle Dataset / Custom Dataset
- **Split**: 70% Training, 15% Validation, 15% Test
- **Image Size**: 128×128 pixels
- **Preprocessing**: Normalization, Resizing, Augmentation

## Project Structure
\`\`\`
Project_<Number>_<TeamNumber>/
├── code/
│   ├── train.py          # Training script
│   ├── model.py          # CNN architecture
│   ├── dataset.py        # Data handling
│   ├── evaluate.py       # Evaluation metrics
│   ├── utils.py          # Utility functions
│   └── requirements.txt   # Dependencies
├── saved_model/
│   └── best_model.h5     # Trained model
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── report.pdf            # Final report (8-4 pages)
└── README.md             # This file
\`\`\`

## Installation

1. Create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r code/requirements.txt
\`\`\`

## Running the Project

### Training the Model
\`\`\`bash
python code/train.py
\`\`\`

This will:
- Load and preprocess the dataset
- Train the CNN model with the specified architecture
- Save the best model to `saved_model/best_model.h5`
- Generate training curves (accuracy and loss)

### Evaluating the Model
\`\`\`bash
python code/evaluate.py
\`\`\`

This will:
- Load the trained model
- Generate predictions on the test set
- Calculate evaluation metrics (accuracy, precision, recall, F1-score)
- Generate confusion matrix and sample predictions

## Model Architecture

The CNN consists of 4 convolutional blocks:
- **Block 1**: Conv(32) + MaxPool + Dropout(0.25)
- **Block 2**: Conv(64) + MaxPool + Dropout(0.25)
- **Block 3**: Conv(128) + MaxPool + Dropout(0.25)
- **Block 4**: Conv(256) + MaxPool + Dropout(0.25)
- **Dense Layers**: 512 neurons + Dropout(0.5) → Output layer with Softmax

## Training Configuration

- **Batch Size**: 32
- **Epochs**: 40
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Callbacks**: EarlyStopping, ModelCheckpoint

## Expected Results

- **Test Accuracy**: 85-95%
- **F1-Score**: 0.85-0.95
- **Precision/Recall**: Varies by denomination

## Key Features

1. **Data Augmentation**: Rotation, translation, zoom, brightness adjustments
2. **Regularization**: Dropout layers to prevent overfitting
3. **Callbacks**: Early stopping for optimal convergence
4. **Comprehensive Evaluation**: Confusion matrix, classification report, sample predictions

## Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Lighting variations | Data augmentation with brightness adjustments |
| Overfitting | Dropout layers and regularization |
| Class imbalance | Data augmentation and class weighting |
| Low accuracy | Transfer Learning with ResNet or EfficientNet |

## Suggested Improvements

1. **Transfer Learning**: Use pre-trained models like ResNet50 or EfficientNet
2. **Ensemble Methods**: Combine multiple models for better accuracy
3. **Real-world Testing**: Test with real banknote images and different lighting conditions
4. **Mobile Deployment**: Convert model to TensorFlow Lite for mobile apps
5. **Advanced Preprocessing**: Multi-scale preprocessing and edge detection

## Authors
Team: <Team 18 >
Project: <Project 8>
