# Shoes-classification-using-CNN-adidas-vs-Nike

This project is a **deep learning image classification model** designed to distinguish between **Nike** and **Adidas** logos using **Convolutional Neural Networks (CNNs)**.

It uses a labeled image dataset of logos to train a binary classifier that can accurately detect whether a given image contains a **Nike** or **Adidas** logo.



### 🚀 Technologies Used

* Python 🐍
* NumPy
* Matplotlib
* PIL (Python Imaging Library)
* TensorFlow / Keras (for building CNN)
* tqdm (for progress visualization)


### 🖼️ Data Preparation

* Images are loaded using PIL.
* All images are resized to **120x120 pixels**.
* Images are converted to grayscale (`.convert('L')`) for simplified feature extraction.
* Labels: `0 = Nike`, `1 = Adidas`
* Dataset is shuffled to ensure unbiased training.


### 🧠 Model Architecture

A simple CNN with:

* `Conv2D`, `MaxPooling2D`, `Flatten`
* Followed by Dense layers for classification

**Output Layer:**

* `Dense(1, activation='sigmoid')` → for binary classification

**Loss Function:**

* `binary_crossentropy`


### 📈 Training

The model is trained using:

* Optimizer: `Adam`
* Epochs: 10 (configurable)
* Normalized input (`pixel value / 255.0`)
* Validation can be added using `train_test_split`


### ✅ Results

* Accuracy improves over epochs
* CNN generalizes well for logo detection
* Final trained model can be saved and used for inference on new images


### 🧪 Example Usage

```python
model.predict(new_image.reshape(-1, 120, 120, 1))
# Output: [0] → Nike, [1] → Adidas
```
### 📌 To Run This Project

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/nike-vs-adidas-logo-classification.git
   cd nike-vs-adidas-logo-classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place images in the correct folders:

   ```
   /train/Nike/
   /train/Adidas/
   /test/
   ```

4. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook deeplearning_imagedetection_project.ipynb
   ```

### 📷 Sample Predictions

| Input Image                  | Predicted Label |
| ---------------------------- | --------------- |
| ![Nike](sample_nike.jpg)     | Nike            |
| ![Adidas](sample_adidas.jpg) | Adidas          |


