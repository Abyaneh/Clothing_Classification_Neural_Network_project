# Clothing Classification


**Objective**: Classify clothing images into categories such as shirts, pants, and dresses using CNN models.

### Data Characteristics:
- **Dataset**: 70,000 images of clothing, divided into 60,000 images for training and validation, and 10,000 for testing.
- **Challenges**:
  - Designed three different CNN models with varying architectures.
  - Experimented with different hyperparameters (epochs, batch size, dropout layers) for model optimization.

### Model 1 Architecture:
- 4 convolutional layers with filter sizes of 32, 64, 128, and 256, each followed by **MaxPooling** layers.
- The final layers include **Flatten**, **Dense (128 neurons)** with ReLU, and an output layer with **Softmax** for multi-class classification.

### Results:
- **Model 1 Test Accuracy**: 91.35%
- **Model 2 Test Accuracy**: 87.83%
- **Model 3 Test Accuracy**: 92.17%
- **Best Model**: Model 3, which included a **Dropout Layer (0.5)** to prevent overfitting, achieved the best performance with a test accuracy of **92.17%**.

#### Model1
![Model1](https://github.com/Abyaneh/Neural_Network_projects/blob/main/Photos/Epoch-Accuracy_and_Epoch_Loss_Chart_project3_10epochs_model1.png)

#### Model2
![Model2](https://github.com/Abyaneh/Neural_Network_projects/blob/main/Photos/Epoch-Accuracy_and_Epoch_Loss_Chart_project3_10epochs_model2.png)

#### Model3
![Model3](https://github.com/Abyaneh/Neural_Network_projects/blob/main/Photos/Epoch-Accuracy_and_Epoch_Loss_Chart_project3_10epochs_model3.png)

### Analysis of Performance:
- Models were evaluated using accuracy and loss graphs. Model 3, with its dropout layer, showed the best balance between training and validation accuracy, reducing overfitting compared to the other models.


[Back to Top](#table-of-contents)
