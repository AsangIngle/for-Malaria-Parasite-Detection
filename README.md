**Malaria Cell Classification using CNN + Discrete Wavelet Transform (DWT)**

**Overview**
This project implements a custom CNN architecture integrated with Discrete Wavelet Transform (DWT) for binary classification of malaria cell images into Parasitized and Uninfected classes.
Instead of relying only on spatial convolutions, the model explicitly decomposes feature maps into frequency sub-bands (LL, LH, HL, HH) using a custom DWT layer. This helps capture fine-grained texture patterns that are important in microscopic blood smear images.

The model is trained and evaluated on a balanced malaria dataset using TensorFlow/Keras.

**Key Highlights**
-Custom DWT Layer implemented from scratch (no external wavelet libraries)
-Multi-stage DWT + CNN feature extraction
-Strong regularization using BatchNorm and Dropout
-End-to-end training, validation, and testing pipeline
-Detailed evaluation using Accuracy, Precision, Recall, F1-score, MCC, and Confusion Matrix

**Dataset**
-Classes:
  -Parasitized
  -Uninfected
-Image Size: 50 × 50 × 3
-Total Images: 5512
  -Parasitized: 2756
  -Uninfected: 2756
Directory Structure:
 Asang_splited_data/
 └── test/
    ├── Parasitized/
    └── Uninfected/

⚠️ Note: In this implementation, the same directory is used for train, validation (via split), and test. This is not ideal for real research but kept as-is to reflect the current code.
    
**Model Architecture**
 **Input**
   Shape: (50, 50, 3)
**Feature Extraction**
  -Custom DWTLayer (downsamples spatially, expands channels ×4)
  -Stacked Conv2D + BatchNorm + LeakyReLU
  -Three levels of DWT decomposition
**Classifier**
 -Flatten
 -Dense layers: 1024 → 512 → 128
 -Dropout: 0.5
 -Output: Dense(2) + Softmax
**Model Size**
 -Total Parameters: ~6.03M
 -Trainable Parameters: ~6.02M

**DWT Layer Explanation**
The custom DWTLayer:
 -Splits feature maps into even/odd rows and columns
 -Computes:
   -LL (approximation)
   -HL, LH, HH (detail coefficients)
-Concatenates all sub-bands along the channel dimension
This mimics a Haar Wavelet Transform and enables frequency-aware learning inside the CNN.

**Training Configuration**
-Optimizer: Adam (lr = 0.0005)
-Loss: Categorical Cross-Entropy
-Batch Size: 64
-Epochs: up to 50
-Callbacks:
  -EarlyStopping (patience = 20)
  -ModelCheckpoint (best weights only)
  -Custom Test Accuracy Callback (evaluates test set after each epoch) 

**Data Augmentation**
 -Rotation
 -Width & height shift
 -Shear
 -Zoom
 -Horizontal flip

 Evaluation Metrics (Test Set)
Metric                  	Value
Accuracy	                96.61%
Precision (weighted)	    96.62%
Recall (weighted)	        96.61%
F1-Score (weighted)	      96.61%
MCC	                      0.9323

**Outputs Saved**
 -Trained weights (.weights.h5)
 -Accuracy vs Epoch graph
 -Loss vs Epoch graph
 -Confusion Matrix image
All outputs are saved to Google Drive.

