import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("Housing.csv")

print("Dataset loaded successfully!")
print()


# ============================================================
# 2. DISPLAY DATASET
# ============================================================

print("First 5 rows:")
print(data.head())

print()
print("Dataset shape:")
print(data.shape)


# ============================================================
# 3. CHECK FOR NULL VALUES
# ============================================================

print()
print("Checking for null values:")
print(data.isnull().sum())


# ============================================================
# 4. REMOVE NULL VALUES
# ============================================================

# This will remove rows containing missing values.
# Your current dataset has no null values,
# so no rows will actually be removed.

data = data.dropna()

print()
print("Dataset after removing null values:")
print(data.shape)


# ============================================================
# 5. SELECT INPUT AND OUTPUT
# ============================================================

# Input = area of the house
X = data[["area"]]

# Output = price of the house
y = data["price"]


# ============================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print()
print("Training data:", len(X_train))
print("Testing data:", len(X_test))


# ============================================================
# 7. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print()
print("Model trained successfully!")


# ============================================================
# 9. GET SLOPE AND INTERCEPT
# ============================================================

slope = model.coef_[0]
intercept = model.intercept_

print()
print("========== MODEL INFORMATION ==========")
print("Slope:", slope)
print("Intercept:", intercept)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 11. COMPARE ACTUAL AND PREDICTED VALUES
# ============================================================

result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print()
print("========== ACTUAL VS PREDICTED ==========")
print(result.head(10))


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print()
print("========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)


# ============================================================
# 13. PREDICT PRICE FOR NEW HOUSE
# ============================================================

print()
print("========== HOUSE PRICE PREDICTION ==========")

area = float(input("Enter house area: "))

prediction = model.predict([[area]])

print()
print("House area:", area)
print("Predicted house price:", prediction[0])


# ============================================================
# 14. PLOT THE DATA AND REGRESSION LINE
# ============================================================

plt.scatter(
    X,
    y,
    label="Actual Houses"
)

# Sort the area values so that the regression line
# is drawn correctly.
sorted_data = data.sort_values("area")

X_sorted = sorted_data[["area"]]

y_line = model.predict(X_sorted)

plt.plot(
    X_sorted,
    y_line,
    label="Regression Line"
)

plt.xlabel("House Area")
plt.ylabel("House Price")

plt.title("Linear Regression: Area vs House Price")

plt.legend()

plt.show()
