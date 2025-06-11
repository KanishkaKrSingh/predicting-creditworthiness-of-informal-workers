import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load data
df = pd.read_csv("data/processed/user_income_features_with_labels.csv")

# Feature columns
features = [
    'predicted_income', 'female_literacy', 'male_literacy',
    'distance_from_city', 'gdp_per_capita', 'upi_txn_volume'
]

# Encode categorical features
for col in ['gender', 'rural_urban', 'work_sector']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    joblib.dump(le, f"models/le_{col}.pkl")
    features.append(col)

X = df[features]
y = to_categorical(df["repayment_category"], num_classes=5)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/stage2_scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')  # 5 repayment classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stop])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save model
model.save("models/stage2_final_model.keras")
print("✅ Stage-2 repayment classifier saved.")
