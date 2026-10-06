import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# =====================================================
# 1. IMPORT CSV
# =====================================================

data = pd.read_csv("student_data.csv")

print("====================================")
print("          STUDENT DATASET")
print("====================================")

print(data.head())


# =====================================================
# 2. CREATE REQUIRED COLUMNS
# =====================================================

# Attendance percentage
# Assuming 100 working days
data["Attendance"] = 100 - data["absences"]


# Create Internal Mark from study-related variables
# Scaled to approximately 0-50
data["Internal Mark"] = (
    (data["studytime"] / data["studytime"].max()) * 25
    +
    (data["Medu"] / data["Medu"].max()) * 25
)


# Create Final Mark from internal-related factors
# Scaled to approximately 0-100
data["Final Mark"] = (
    data["Internal Mark"] * 0.6
    +
    (data["Attendance"] / 100) * 40
)


# =====================================================
# 3. STATISTICAL MEASURES
# =====================================================

print("\n====================================")
print("              OUTPUT")
print("====================================")


# ---------------- FINAL MARK ----------------

print("\nMean of final Mark =",
      round(data["Final Mark"].mean(), 3))

print("Mode of final Mark =",
      round(data["Final Mark"].mode().iloc[0], 3))

print("Median of final Mark =",
      round(data["Final Mark"].median(), 3))

print("Standard deviation of final Mark =",
      round(data["Final Mark"].std(), 3))


# ---------------- INTERNAL MARK ----------------

print("\nMean of Internal Marks =",
      round(data["Internal Mark"].mean(), 3))

print("Mode of Internal Marks =",
      round(data["Internal Mark"].mode().iloc[0], 3))

print("Median of Internal Marks =",
      round(data["Internal Mark"].median(), 3))

print("Standard deviation of Internal Marks =",
      round(data["Internal Mark"].std(), 3))


# ---------------- ATTENDANCE ----------------

print("\nMean of Attendance =",
      round(data["Attendance"].mean(), 3))

print("Median of Attendance =",
      round(data["Attendance"].median(), 3))

print("Mode of Attendance =",
      round(data["Attendance"].mode().iloc[0], 3))

print("Standard deviation of Attendance =",
      round(data["Attendance"].std(), 3))


# =====================================================
# 4. CLASSIFICATION DATA
# =====================================================

X = data[
    [
        "Internal Mark",
        "Final Mark",
        "Attendance"
    ]
]

y = data["passed"]


# =====================================================
# 5. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 6. STANDARDIZATION
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# =====================================================
# 7. BAYESIAN CLASSIFICATION
# =====================================================

bayes = GaussianNB()

bayes.fit(X_train, y_train)

bayes_pred = bayes.predict(X_test)

bayes_accuracy = accuracy_score(
    y_test,
    bayes_pred
)


print("\n====================================")
print("      BAYESIAN CLASSIFICATION")
print("====================================")

print("Actual Values =", y_test.values)

print("Predicted Values =", bayes_pred)

print("Accuracy =",
      round(bayes_accuracy, 4))


# =====================================================
# 8. SVM CLASSIFICATION
# =====================================================

svm = SVC(kernel="linear")

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)


print("\n====================================")
print("          SVM CLASSIFICATION")
print("====================================")

print("Actual Values =", y_test.values)

print("Predicted Values =", svm_pred)

print("Accuracy =",
      round(svm_accuracy, 4))


# =====================================================
# 9. MODEL COMPARISON
# =====================================================

print("\n====================================")
print("          MODEL COMPARISON")
print("====================================")

print("Bayesian Accuracy =",
      round(bayes_accuracy, 4))

print("SVM Accuracy =",
      round(svm_accuracy, 4))


print("\n====================================")
print("             COMPLETED")
print("====================================")