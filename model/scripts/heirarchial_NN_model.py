import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# Paths
data_dir = r"D:\Python\Credit_UnderWriting_Model\data\processed"
model_dir = r"D:\Python\Credit_UnderWriting_Model\models"
os.makedirs(model_dir, exist_ok=True)

# Load Stage-1 predictions and true incomes
person_path = os.path.join(data_dir, "synthetic_person_data.csv")
df = pd.read_csv(person_path)

# 1. Define repayment capability categories based on predicted_income
# Use quantile-based bins for balanced classes
labels = [0, 1, 2, 3, 4]
df['repay_cat'] = pd.qcut(df['predicted_income'], q=5, labels=labels).astype(int)

# 2. One-hot encode categorical features
categorical_cols = ['work_sector', 'area_type', 'gender']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# 3. Drop rows with missing values in features or target
exclude_cols = ['repay_cat', 'predicted_income', 'district', 'state']
features = [col for col in df.columns if col not in exclude_cols]
df = df.dropna(subset=features + ['repay_cat'])

# 4. Convert all feature columns to float32 explicitly
df[features] = df[features].astype(np.float32)

# 5. Prepare features and target arrays with correct dtypes
X = df[features].values.astype(np.float32)
y = to_categorical(df['repay_cat'], num_classes=5).astype(np.float32)

# 6. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025, stratify=df['repay_cat']
)

# 7. Scale numeric features only
num_features = ['gdp_per_capita', 'literacy_rate_male', 'literacy_rate_female', 'avg_upi_txn_volume', 'distance_from_city']
num_idx = [features.index(f) for f in num_features]
scaler = StandardScaler()
X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

# Save scaler for later use
joblib.dump(scaler, os.path.join(model_dir, 'stage2_scaler.pkl'))

# 8. Build improved classification model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),      # 4 front hidedn layers
    Dropout(0.3),
    Dense(64),
    BatchNormalization(),    # 2 middle hidden layers
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(32),
    BatchNormalization(),    # hidden layer
    LeakyReLU(alpha=0.1),
    Dense(5, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 9. Callbacks with increased patience
es = EarlyStopping(patience=20, restore_best_weights=True)
mc = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'best_stage2_model.h5'),
    save_best_only=True, monitor='val_loss'
)

# 10. Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[es, mc]
)

# 11. Evaluate on test set
eval_results = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_results[0]:.4f}, Test Accuracy: {eval_results[1]:.4f}")

# 12. Save final model
model.save(os.path.join(model_dir, 'final_stage2_model.h5'))
print(f"✅ Stage-2 model and artifacts saved at {model_dir}")
