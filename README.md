# Fruit Classification (10 Classes)

This project implements a deep learning pipeline for fruit image classification using TensorFlow and Transfer Learning with MobileNetV2 and Xception.

# Dataset
Source: [Kaggle - Fruit Classification (10 Classes)](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)  
- Split: Train (80%), Validation (20%), Test set provided separately.

# Workflow
1. **Data Loading & Visualization**  
     Uses `image_dataset_from_directory` with train/val split.  
     Displays sample images with class labels.

2. **Preprocessing**  
     Normalization (`x/255.0`).  
     Augmentation: Random Flip, Rotation, Zoom.

3. **Model Building**  
     Transfer Learning with **MobileNetV2** (frozen layers).  
     Dense layers for classification.  
     Output: 10-class softmax.

4. **Training & Evaluation**  
     Optimizer: Adam  
     Loss: Sparse Categorical Crossentropy  
     Metrics: Accuracy, Precision, Recall  
     Training history plotted (loss & accuracy curves).

5. **Prediction**  
     Load image with OpenCV.  
     Resize & scale.  
     Predict class using trained model.  
    Output predicted fruit name.

# How to Run
```bash
pip install tensorflow matplotlib opendatasets opencv-python numpy
python fruit_classification.py