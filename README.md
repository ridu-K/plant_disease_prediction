# 🌿 Plant Disease Prediction Using Convolutional Neural Networks (CNN)

This project implements a **deep learning–based plant disease classification system** using a **Convolutional Neural Network (CNN)** trained on the **PlantVillage dataset**. The model predicts plant diseases from leaf images with high accuracy and supports real-time image-based inference.

---

## 📌 Project Overview

* 📷 **Input**: Leaf images of crops
* 🧠 **Model**: Custom CNN built using TensorFlow & Keras
* 🏷️ **Classes**: 38 plant disease categories
* 📊 **Validation Accuracy**: **~86.9%**
* 🧪 **Dataset**: PlantVillage (Kaggle)
* 🖥️ **Platform**: Google Colab

---

## 🚀 Key Features

* ✅ Reproducible training using fixed random seeds
* 🧠 CNN-based image classification
* 📂 Automatic dataset loading using `ImageDataGenerator`
* 📊 Training & validation performance visualization
* 🖼️ Image upload and real-time disease prediction
* 💾 Model & class label persistence

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy, Pandas**
* **Matplotlib**
* **Pillow (PIL)**
* **Kaggle API**
* **Google Colab**

---

## 📂 Dataset

**PlantVillage Dataset** (Color Images)

* 📌 Source: Kaggle
  [https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* 🏷️ Total Classes: **38**
* 🖼️ Image Size: **256 × 256 (RGB)**

Dataset structure used:

```
plantvillage dataset/
├── color/
├── grayscale/
└── segmented/
```

Only the **color images** are used for training.

---

## 🔁 Reproducibility

To ensure consistent results, random seeds are fixed for:

* Python `random`
* NumPy
* TensorFlow

```python
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
```

---

## ⚙️ Data Preprocessing

* Images resized to **224 × 224**
* Pixel values normalized to **[0, 1]**
* Dataset split:

  * **80% Training**
  * **20% Validation**

Using `ImageDataGenerator`:

* Rescaling
* Automatic class labeling
* Efficient batch loading

---

## 🧠 Model Architecture

| Layer      | Description         |
| ---------- | ------------------- |
| Conv2D     | 32 filters, ReLU    |
| MaxPooling | 2×2                 |
| Conv2D     | 64 filters, ReLU    |
| MaxPooling | 2×2                 |
| Flatten    | Feature vector      |
| Dense      | 256 neurons, ReLU   |
| Output     | 38 neurons, Softmax |

**Total Parameters**: ~47.8 million

---

## 🏋️ Model Training

* Optimizer: **Adam**
* Loss Function: **Categorical Crossentropy**
* Epochs: **5**
* Batch Size: **32**

### Training Results

* **Training Accuracy**: ~97.7%
* **Validation Accuracy**: **86.92%**

---

## 📈 Performance Visualization

* Training vs Validation **Accuracy**
* Training vs Validation **Loss**

Graphs are plotted using `matplotlib` for performance analysis.

---

## 🔍 Model Evaluation

```text
Validation Accuracy: 86.92%
```

The model generalizes well across unseen plant disease images.

---

## 🔮 Prediction System

### Supported Input

* Upload leaf images (`.jpg`, `.png`)
* Automatic preprocessing
* Real-time disease prediction

### Example Predictions

| Image       | Prediction                          |
| ----------- | ----------------------------------- |
| Apple Leaf  | Apple___Black_rot                   |
| Corn Leaf   | Corn_(maize)___Cercospora_leaf_spot |
| Orange Leaf | Orange___Haunglongbing              |

---

## 💾 Model & Metadata Saving

* Trained model saved as:

  ```
  plant_disease_prediction_model.h5
  ```
* Class labels stored as:

  ```
  class_indices.json
  ```

> ⚠️ Note: `.h5` format is legacy; `.keras` format is recommended for future use.

---

## ▶️ How to Run the Project

1. Open **Google Colab**
2. Upload `kaggle.json`
3. Install Kaggle API
4. Download & extract dataset
5. Run cells sequentially
6. Train the CNN model
7. Upload an image to predict disease

---

## 📌 Applications

* Smart agriculture systems
* Early disease detection
* Crop monitoring
* Precision farming
* Agricultural decision support systems

---

## 🔮 Future Enhancements

* Use **Transfer Learning** (ResNet, MobileNet, EfficientNet)
* Deploy using **Streamlit / Flask**
* Mobile app integration
* Improve accuracy with data augmentation
* Convert model to `.keras` format

