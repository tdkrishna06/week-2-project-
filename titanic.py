# TITANIC SURVIVAL PREDICTION
# USING LOGISTIC REGRESSION
# ============================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ============================================
# 2. Load the dataset
# ============================================

df = pd.read_csv("tested.csv")


# ============================================
# 3. Display the dataset
# ============================================

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical information:")
print(df.describe())


# ============================================
# 4. Check missing values
# ============================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================
# 5. Select useful columns
# ============================================

# We don't need PassengerId, Name, Ticket and Cabin
# for this simple Logistic Regression model.

df = df[
    [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]


# ============================================
# 6. Handle missing values
# ============================================

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Fare with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Fill missing Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================
# 7. Convert categorical data into numbers
# ============================================

# Sex:
# male   -> 0
# female -> 1

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})


# Embarked:
# S -> 0
# C -> 1
# Q -> 2

df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})


print("\nDataset after converting text to numbers:")
print(df.head())


# ============================================
# 8. Separate features and target
# ============================================

# X = input features
# y = output/target

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]

y = df["Survived"]


print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# ============================================
# 9. Split data into training and testing
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 10. Create Logistic Regression model
# ============================================

model = LogisticRegression(max_iter=1000)


# ============================================
# 11. Train the model
# ============================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================
# 12. Make predictions
# ============================================

y_pred = model.predict(X_test)


print("\nPredicted values:")
print(y_pred)


# ============================================
# 13. Calculate accuracy
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy percentage:")
print(accuracy * 100, "%")


# ============================================
# 14. Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# 15. Classification Report
# ============================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# 16. Predict a new passenger
# ============================================

# Example passenger:
#
# Pclass   = 3
# Sex      = 0 (male)
# Age      = 25
# SibSp    = 0
# Parch    = 0
# Fare     = 10
# Embarked = 0 (S)

new_passenger = [[
    3,
    0,
    25,
    0,
    0,
    10,
    0
]]


prediction = model.predict(new_passenger)


print("\nNew Passenger Prediction:")

if prediction[0] == 1:
    print("Passenger is predicted to SURVIVE.")
else:
    print("Passenger is predicted NOT TO SURVIVE.")


# ============================================
# 17. Prediction probability
# ============================================

probability = model.predict_proba(new_passenger)

print("\nPrediction Probability:")
print("Probability of Not Surviving:", probability[0][0])
print("Probability of Surviving:", probability[0][1])