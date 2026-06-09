import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("diabetes.csv")

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ==============================
# VISUALIZATION
# ==============================
plt.figure(figsize=(6, 4))
df['Outcome'].value_counts().plot(kind='bar')
plt.title("Diabetes Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

# ==============================
# FEATURES AND TARGET
# ==============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# TRAIN MODEL
# ==============================
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# ==============================
# TEST MODEL
# ==============================
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# ==============================
# USER PREDICTION
# ==============================
print("\n===== DIABETES PREDICTION SYSTEM =====")

input_data = pd.DataFrame(
   [[6, 148, 72, 35, 0, 33.6, 0.627, 50]], #Diabetic
  #  [[2, 90, 68, 25, 70, 25.0, 0.250, 28]],     #Non Diabetic
    columns=X.columns
)

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)

if prediction[0] == 0:
    print("\nResult: Person is NOT Diabetic")
else:
    print("\nResult: Person is Diabetic")