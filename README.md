# Diabetes-Prediction-using-Machine-Learning
Machine Learning project that predicts diabetes using Logistic Regression, Python, Pandas, and Scikit-Learn based on patient health data.
## 📌 Project Overview

This project aims to predict whether a person is diabetic or non-diabetic based on various medical attributes using Machine Learning. The model is trained on the Pima Indians Diabetes Dataset and uses Logistic Regression for classification.

Developed as part of the **AI/ML Internship Program at InternPe**.

---

## 🎯 Objective

The primary objective of this project is to build a Machine Learning model that can assist in predicting diabetes based on patient health data and provide quick preliminary assessments.

---

## 📊 Dataset Information

The dataset contains 768 patient records with the following attributes:

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| Pregnancies              | Number of times pregnant                         |
| Glucose                  | Plasma glucose concentration                     |
| BloodPressure            | Diastolic blood pressure (mm Hg)                 |
| SkinThickness            | Triceps skin fold thickness (mm)                 |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                   |
| BMI                      | Body Mass Index                                  |
| DiabetesPedigreeFunction | Diabetes hereditary score                        |
| Age                      | Age of the patient                               |
| Outcome                  | Target Variable (0 = Non-Diabetic, 1 = Diabetic) |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* VS Code

---

## 🔄 Project Workflow

### 1. Data Collection

* Load the diabetes dataset using Pandas.

### 2. Data Exploration

* Analyze dataset structure.
* View statistical summaries.
* Understand feature distributions.

### 3. Data Preprocessing

* Separate features and target variable.
* Split dataset into training and testing sets.
* Apply feature scaling using StandardScaler.

### 4. Model Training

* Train a Logistic Regression model using training data.

### 5. Model Evaluation

* Calculate Accuracy Score.
* Generate Confusion Matrix.
* Display Classification Report.

### 6. Prediction System

* Accept new patient data.
* Predict whether the patient is diabetic or non-diabetic.

---

## 🤖 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a supervised machine learning classification algorithm used to predict categorical outcomes.

In this project:

* Output = 0 → Non-Diabetic
* Output = 1 → Diabetic

---

## 📈 Results

The model successfully predicts diabetes using medical attributes.

Evaluation metrics used:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 🚀 How to Run the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/Diabetes-Prediction-ML.git
```

### Navigate to Project Folder

```bash
cd Diabetes-Prediction-ML
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Run Project

```bash
python diabetes_prediction.py
```

---

## 📂 Project Structure

```text
Diabetes-Prediction-ML
│
├── diabetes.csv
├── diabetes_prediction.py
├── README.md
└── requirements.txt
```

---

## 📚 Learning Outcomes

Through this project, I gained hands-on experience in:

* Data Preprocessing
* Machine Learning Model Training
* Logistic Regression
* Feature Scaling
* Model Evaluation
* Healthcare Data Analysis
* Python Programming

---

## 🙏 Acknowledgement

This project was completed as part of the AI/ML Internship Program at InternPe. Special thanks to InternPe for providing practical exposure to Machine Learning and Data Science concepts.

---

## 👨‍💻 Author

**Shaik Asif**

AI/ML Intern | Machine Learning Enthusiast | Python Developer

LinkedIn: www.linkedin.com/in/shaikasif369
