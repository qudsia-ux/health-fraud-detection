import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("fraud_dataset.csv")

# One-hot encoding
data = pd.get_dummies(data, columns=["Diagnosis"])

# Split features & target
X = data.drop("Is_Fraudulent", axis=1)
y = data["Is_Fraudulent"]

# Save columns
pickle.dump(X.columns, open("columns.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# PREPROCESSING
# -------------------------------
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

# Save preprocessors
pickle.dump(imputer, open("imputer.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# -------------------------------
# TRAIN MODELS (VERY IMPORTANT)
# -------------------------------

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Decision Tree ✅ FIX HERE
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# -------------------------------
# SAVE MODELS
# -------------------------------
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(dt_model, open("dt_model.pkl", "wb"))

print("✅ All models trained and saved successfully!")