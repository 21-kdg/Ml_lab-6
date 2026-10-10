# ML Practical 6: Artificial Neural Network (ANN) Classification

## Student Details
- **Name:** Krishna Gupta
- **Roll Number:** 10
- **Section-Batch:** A-A1
- **Experiment:** Lab 6

## 1. Aim
To implement an Artificial Neural Network (ANN) using TensorFlow and Keras to classify seed samples into three different classes based on their geometric characteristics.

## 2. Dataset Description
The dataset contains 210 records and 10 columns, including the target class and two additional columns named `Unnamed: 8` and `Unnamed: 9`.

The seven intended input features are:
- Area
- Perimeter
- Compactness
- Length of kernel
- Width of kernel
- Asymmetry coefficient
- Length of kernel groove

**Target variable:** `Class (1, 2, 3)`

The target contains three classes: 1, 2, and 3. These are converted into labels 0, 1, and 2 for ANN training.

## 3. Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- Google Colab / Jupyter Notebook

## 4. Methodology
1. Import the required libraries for data processing, visualization, model development, and evaluation.
2. Load the seed dataset and inspect its shape, columns, data types, statistical summaries, missing values, and duplicate records.
3. Separate the input features from the target variable.
4. Convert the original target labels 1, 2, and 3 into encoded labels 0, 1, and 2.
5. Divide the dataset into 80% training data and 20% testing data using `train_test_split()` with `random_state=42` and stratified sampling.
6. Apply `StandardScaler` to standardize the training and testing features.
7. Construct a sequential ANN with two hidden layers containing 16 and 8 neurons, respectively.
8. Use ReLU activation in the hidden layers and Softmax activation in the output layer.
9. Compile the model using the Adam optimizer, a learning rate of 0.001, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
10. Train the network for up to 100 epochs with a batch size of 16 and a validation split of 20%.
11. Apply EarlyStopping with a patience of 25 and restore the best weights based on validation loss.
12. Plot training and validation accuracy and loss to observe the learning process.
13. Evaluate the trained model on the test dataset.
14. Generate class predictions using the output probabilities and convert the encoded predictions back to the original class labels.
15. Evaluate the classifier using accuracy, weighted precision, weighted recall, weighted F1-score, a classification report, and confusion matrices.

## 5. ANN Model Architecture
The neural network uses the following architecture:

| Layer | Configuration |
|---|---|
| Input layer | 7 input features intended |
| Hidden layer 1 | 16 neurons, ReLU |
| Hidden layer 2 | 8 neurons, ReLU |
| Output layer | 3 neurons, Softmax |

### Training Configuration
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** Sparse categorical cross-entropy
- **Maximum epochs:** 100
- **Batch size:** 16
- **Validation split:** 20%
- **Early stopping patience:** 25

## 6. Results

The notebook reports the following test performance:

| Evaluation Metric | Score |
|---|---:|
| Test loss | 0.3856 |
| Accuracy | 83.33% |
| Weighted precision | 84.16% |
| Weighted recall | 83.33% |
| Weighted F1-score | 82.18% |

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Class 1 | 0.89 | 0.57 | 0.70 |
| Class 2 | 0.81 | 0.93 | 0.87 |
| Class 3 | 0.82 | 1.00 | 0.90 |

### Confusion Matrix

The confusion matrix reported in the notebook is:

| Actual / Predicted | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|
| Class 1 | 8 | 3 | 3 |
| Class 2 | 1 | 13 | 0 |
| Class 3 | 0 | 0 | 14 |

The model correctly classified 35 out of 42 test samples, resulting in an accuracy of 83.33%.

## 7. Visualizations
The notebook includes:
- Training and validation accuracy curves.
- Training and validation loss curves.
- A confusion matrix showing classification results.
- A percentage-based confusion matrix.
- A comparison of actual and predicted class labels.

These visualizations help evaluate learning progress, classification performance, and errors across the three classes.

## 8. Conclusion
An Artificial Neural Network was implemented using TensorFlow and Keras to classify seed samples into three classes. The experiment included feature scaling, label encoding, model training, early stopping, and performance evaluation. The model achieved a test accuracy of 83.33%, demonstrating its ability to classify the samples based on their measured features.

## 9. How to Run
1. Open `lab6.ipynb` in Google Colab or Jupyter Notebook.
2. Upload the dataset required by the notebook.
3. Install the required libraries if they are not already available.
4. Run the notebook cells in sequence.
5. Review the training curves, classification report, confusion matrix, and final evaluation metrics.

## 10. Files
- `lab6.ipynb` – Notebook containing preprocessing, ANN construction, training, prediction, and evaluation.
- Dataset file – The seed dataset used by the notebook.

