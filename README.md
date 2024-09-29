# Landmark Classification & Tagging for Social Media
This project was done as a part of the AWS AI/Ml Scholarship Program's Advanced Nanodegree.

The goal of the project is to predict the location of an image based on landmarks depicted within it, even when metadata is missing. 

Photo sharing and storage services like to have location data for each photo that is uploaded. This helps them offer advanced features like automatic photo tagging and organization, improving the overall user experience. However, photos often lack location metadata due to the absence of GPS data or privacy concerns. This project tackles this issue by classifying landmarks to infer location data.

## Project Overview

1. **Created a CNN to Classify Landmarks (from Scratch)**  
   - Visualized and preprocessed the dataset for training.
   - Built a convolutional neural network from scratch to classify the landmarks.
   - Exported the best network using Torch Script.
   
2. **Created a CNN to Classify Landmarks (using Transfer Learning)**  
   - Investigated different pre-trained models and select one for classification.
   - Trained and tested the model, then exported the best transfer-learned network using Torch Script.

3. **Deploy Your Algorithm in an App**  
   - Used the best model to create an app that allows users to classify landmarks in images.
   - Tested and evaluated the app, reflecting on the model's strengths and weaknesses.


## Project Steps

### 1. Created a CNN from Scratch

- Visualized the dataset and preprocess images.
- Defined a CNN architecture, loss function, and optimizer.
- Trained and validated the model.
- Tested the model to ensure at least 50% accuracy.
- Exported the model using Torch Script.

**File**: `cnn_from_scratch.ipynb`  
**Key Script**: `src/model.py`

### 2. Transfer Learning for Landmark Classification

- Freezed parameters of a pre-trained model and add a final linear layer for landmark classification.
- Trained, validated, and tested the model.
- Tested the model to ensure at least 60% accuracy.
- Exported the model using Torch Script.

**File**: `transfer_learning.ipynb`  
**Key Script**: `src/transfer.py`

### 3. Deployed the Best Model in an App

- Loaded the TorchScript-exported model.
- Built a simple app to classify landmarks in new images.
- Displayed the classification results for an image not in the training or test sets.

**File**: `app.ipynb`

## Key Scripts

### Data Preprocessing
- **File**: `src/data.py`
- **Functions**: `get_data_loaders`, `visualize_one_batch`

### Model Definition
- **File**: `src/model.py`
- **Functions**: Defines the CNN architecture from scratch.

### Transfer Learning
- **File**: `src/transfer.py`
- **Functions**: Defines transfer learning architecture and loading pre-trained models.

### Training and Optimization
- **File**: `src/optimization.py`
- **Functions**: Implements optimizers and schedulers.

### Prediction
- **File**: `src/predictor.py`
- **Functions**: Implements TorchScript-based predictions.

## Conclusion

In this project, I have done data preprocessing, model training, transfer learning, and app deployment. This has helped me gain a deeper understanding of the end-to-end machine learning pipeline.
