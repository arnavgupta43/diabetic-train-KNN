# K-Nearest Neighbors (KNN) - Diabetes Prediction

## 📌 Overview

This project implements a **K-Nearest Neighbors (KNN) classifier** to predict diabetes based on a medical dataset. The dataset consists of patient information, including age, glucose level, BMI, and other health indicators. The model predicts whether a person is diabetic or not.

## 🔥 Features

- **Data Preprocessing:** Handling missing values, feature scaling
- **Model Implementation:** KNN classifier using Scikit-Learn
- **Hyperparameter Tuning:** Optimizing the value of `k`
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

## 📁 Dataset

We use the **Pima Indians Diabetes Dataset**, available on Kaggle/UCI Repository.

## 🛠️ Installation & Setup

Make sure you have Python installed. Then, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/arnavgupta43/diabetic-train-KNN.git
   cd diabetic-train-KNN
   ```
2. Run the script:
   ```bash
   python knn_diabetes.py
   ```

## 📜 Code Structure

```
📂 knn-diabetes-prediction
 ├── 📄 knn_diabetes.py       # Main KNN model implementation
 ├── 📄 README.md             # Project Documentation
 ├── 📄 dataset               # Contains CSV dataset
```

## 📊 Model Evaluation

We evaluate the model using:

- **Accuracy Score**: Measures the overall correctness of predictions.
- **Confusion Matrix**: Analyzes True Positives, False Positives, etc.
- **Precision & Recall**: Helps in cases of class imbalance.

## 🔧 Hyperparameter Tuning

- The optimal `k` value is determined using the **Elbow Method**, where the error rate is plotted against different `k` values.
- Distance metrics like **Euclidean and Manhattan** are experimented with to find the best fit.

## 🤖 Future Enhancements

- Implementing **cross-validation** for better generalization.
- Comparing KNN with other ML models like Decision Trees & Logistic Regression.
- Deploying the model as a **web application**.




Summary of the Effects of Changing n, p, and Distance Calculation Type in KNN

Increasing n_neighbors (n)	- Higher n (e.g., 23) → Lower F1 Score & Accuracy (model underfits).
- Lower n (e.g., 11) → Better F1 Score & Accuracy (balance between bias & variance).
- Too small n (e.g., 3) may overfit.
  
Changing p (Minkowski Power Parameter)	- p=1 (Manhattan Distance) → Slightly better F1 Score, similar accuracy.
- p=2 (Euclidean Distance) → Default choice, good performance.
- p>2 (Higher-order Minkowski) → No improvement, behaves like Euclidean.

  
Changing Distance Metric	- Euclidean (p=2) → Standard, works well.
- Manhattan (p=1) → Slightly better F1 Score in high-dimensional data.
- Hamming → Bad for numerical data, poor F1 Score & Accuracy.


Best Observations from Your Results

✔ For best F1 Score: Use Manhattan (p=2), n=11

✔ For best Accuracy: Use Euclidean (p=2), n=11

✔ Hamming Distance is unsuitable for numerical datasets.

✔ Avoid too high n values, as they lead to underfitting.




## 📝 License

This project is open-source and available under the **MIT License**.

---

🙌 **Contributions are welcome!** Feel free to fork the repository and submit a pull request.

