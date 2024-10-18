# Clothing Classification

![Clothing Classification Image](https://github.com/Abyaneh/Clothing_Classification/blob/main/photos/Clothing%20Classification%20Image.png)

## Table of Contents
- [Introduction](#introduction)
- [Data Characteristics](#data-characteristics)
- [Model Architecture](#model-architecture)
- [Results & Performance](#results--performance)
- [Technologies & Tools Used](#technologies--tools-used)
- [How to Run the Project](#how-to-run-the-project)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This project focuses on **classifying clothing images** into different categories using **Convolutional Neural Networks (CNNs)**. The goal is to develop a reliable image classification model capable of categorizing clothing into predefined labels like shirts, pants, and dresses.

## Data Characteristics

- **Dataset**: The dataset consists of **70,000 images** of clothing items, divided into:
  - **60,000 training/validation images**
  - **10,000 test images**
- Each image is labeled with a clothing category such as shirt, pants, etc.
  
The training and validation dataset in [this link](https://drive.google.com/file/d/1XaRAL2GD6z3uL-WxaFBQYLhfwx1M7aqv/view?usp=drive_link)

The test dataset in [this link]()

### Data Preprocessing:
- Images were resized and normalized to ensure consistent input.
- **Data Augmentation** techniques such as rotation, zoom, and flipping were applied to enhance model generalization and prevent overfitting.

## Model Architecture

We developed three CNN models with varying architectures to find the optimal structure for this task.

### Model 1 Architecture:
- 4 convolutional layers with filter sizes of 32, 64, 128, and 256, each followed by **MaxPooling** layers.
- Fully connected layer with 128 neurons and **ReLU** activation.
- Output layer with **Softmax** for multi-class classification.
- **Dropout Layer (0.5)** to mitigate overfitting.

### Model 2 Architecture:
- 3 convolutional layers with filter sizes of 32, 64, and 128, followed by **MaxPooling** layers.
- Fully connected layer and Softmax output layer.

### Model 3 Architecture (Best Performance):
- Deeper architecture with 5 convolutional layers and Dropout, yielding the best test accuracy of **92.17%**.
  
![Model Architecture Diagram](path_to_image_here)

### Optimization Techniques:
- **EarlyStopping** to monitor validation accuracy and halt training when performance plateaus.
- **ModelCheckpoint** to save the best model weights during training.
- **Data Augmentation** for increasing dataset diversity.

## Results & Performance

### Model Comparison:
- **Model 1 Test Accuracy**: 91.35%
- **Model 2 Test Accuracy**: 87.83%
- **Model 3 Test Accuracy**: 92.17% (Best Performing Model)

### Performance Evaluation:
- Accuracy and loss metrics were used for evaluation.
- Model 3 showed the best generalization with minimized overfitting due to the inclusion of **Dropout** and **Data Augmentation**.

#### Model1
![Model1](https://github.com/Abyaneh/Clothing_Classification/blob/main/photos/Epoch-Accuracy_and_Epoch_Loss_Chart_10epochs_model1.png)

#### Model2
![Model2](https://github.com/Abyaneh/Clothing_Classification/blob/main/photos/Epoch-Accuracy_and_Epoch_Loss_Chart_10epochs_model2.png)

#### Model3
![Model3](https://github.com/Abyaneh/Clothing_Classification/blob/main/photos/Epoch-Accuracy_and_Epoch_Loss_Chart_10epochs_model3.png)

## Technologies & Tools Used

- **Programming Language**: Python
- **Libraries**: 
  - **TensorFlow** and **Keras** for deep learning model development.
  - **Numpy** and **Pandas** for data manipulation and preprocessing.
  - **Matplotlib** for visualizing performance metrics like accuracy and loss.
- **Machine Learning Techniques**: 
  - Convolutional Neural Networks (CNNs)
  - Multi-class image classification
- **Optimization Techniques**: EarlyStopping, ModelCheckpoint, Dropout, Data Augmentation

## How to Run the Project

### Step 1: Clone the repository
```bash
git clone https://github.com/Abyaneh/Clothing_Classification/tree/main
```

### Step 3: Train the model
```bash
python train_model.py
```

### Step 4: Evaluate the model
```bash
python evaluate_model.py
```

### Step 5: Test the model on new images
```bash
python test_model.py --image_path /path/to/image
```

## Contributing

Feel free to contribute by submitting a pull request or creating an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
