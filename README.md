# 🌾 Crop Disease Prediction 🦠  

A deep learning-based **crop disease prediction** model with **97% accuracy**, helping farmers detect diseases early using image classification.  

## 📌 Features  
- **High Accuracy (97%)**: Optimized CNN architecture for precise disease detection.  
- **Image-Based Classification**: Uses deep learning to classify diseases from crop images.  
- **Batch Normalization & Dropout**: Improves model stability and reduces overfitting.  
- **Global Average Pooling**: Efficient feature extraction for better performance.  

## 🏗 Model Architecture  
The model is built using **TensorFlow** and **Keras**, featuring multiple **CNN layers**:  

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])
```
## 🛠 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/FalconSlayer51/crop-disease-prediction.git
cd crop-disease-prediction
