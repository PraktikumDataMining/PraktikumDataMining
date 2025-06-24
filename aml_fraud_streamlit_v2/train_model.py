import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("model", exist_ok=True)
df = pd.read_csv("data/Big_Black_Money_Dataset.csv")

# HANYA gunakan kolom yang tersedia
features = ['Transaction Type', 'Amount (USD)', 'Country']
X = df[features]
y = df['Money Laundering Risk Score']

X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model berhasil dilatih dan disimpan.")
