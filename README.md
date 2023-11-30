# CNN Image Classification with Fine-Tuning

This repository contains code for training a Convolutional Neural Network (CNN) for image classification using PyTorch. The code demonstrates the process of fine-tuning models on different datasets, comparing model performance, and implementing Grad-CAM for visualization.

## 1. Setup
- Clone the repository: `git clone https://github.com/yourusername/your-repo.git`
- Navigate to the project folder: `cd your-repo`

## 2. Training AlexNet on CIFAR-10
- Run `mm22mkm.ipynb` to train AlexNet on the CIFAR-10 dataset.
- View training progress and results in the generated CSV file.

## 3. Fine-Tuning Model with Frozen Layers
### Configuration 2: Frozen Base Convolution Blocks
- Load the pre-trained AlexNet model.
- Freeze convolutional layers.
- Modify the last layer for CIFAR-10 classification.
- Train the model with fine-tuned layers.

## 4. Model Comparisons
- Compare the performance of the original AlexNet on CIFAR-10 with the fine-tuned model.
- Display graphs of training and validation accuracy/loss for both models.

## 5. Training AlexNet on TinyImageNet30
- Run `train_alexnet_tiny.py` to train AlexNet on the TinyImageNet30 dataset.
- View training progress and results in the generated CSV file.

## 6. Comparing Results on TinyImageNet30
- Compare the performance of the TinyImageNet30 model with the CIFAR-10 model.
- Display graphs of training and validation accuracy/loss for both models.

## 7. Interpretation of Results
### 7.1 Grad-CAM Visualization
- Install `torchcam`: `pip install torchcam`
- Run `grad_cam_visualization.py` to apply Grad-CAM on correctly and incorrectly classified images.
- View results with overlaid heatmaps on the original images.

### 7.2 Comments on Model Predictions
- Analyze reasons for correct and incorrect predictions based on Grad-CAM visualizations.

### 7.3 Improvement Strategies
- Consider implementing data augmentation, improving model architecture, fine-tuning, dropout, and hyperparameter tuning for better results.

## Conclusion
This repository provides a comprehensive example of CNN image classification, fine-tuning, and result interpretation. Use the provided scripts and adapt them for your specific use case.
