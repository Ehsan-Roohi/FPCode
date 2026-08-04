import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time

# تنظیمات برای فونت نمودارها (مطابق دستور شما)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# ============================================================================
# 1. بارگذاری و فیلتر کردن داده‌ها
# ============================================================================
print("Loading massive cylinder data...")
data = np.load('cylinder_training_data.npz')
X_raw = data['X']
y_raw = data['y']

# انتخاب تصادفی ۵ میلیون نمونه برای مدیریت حافظه
total_samples = X_raw.shape[0]
target_samples = 5000000 
if total_samples > target_samples:
    print(f"Sub-sampling from {total_samples} to {target_samples}...")
    indices = np.random.choice(total_samples, target_samples, replace=False)
    X = X_raw[indices]
    y = y_raw[indices]
else:
    X, y = X_raw, y_raw

del X_raw, y_raw # پاکسازی حافظه

# ============================================================================
# 2. آماده‌سازی داده‌ها
# ============================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

joblib.dump(X_scaler, 'X_scaler_cylinder.joblib')
joblib.dump(y_scaler, 'y_scaler_cylinder.joblib')

# ============================================================================
# 3. ساخت مدل عمیق (4 لایه x 256 نورون - مشابه مدل موفق کوئت شما)
# ============================================================================
model = keras.Sequential([
    layers.Input(shape=(16,), name='input_features'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='linear', name='output_coeffs') # 10 خروجی برای سیلندر
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

# ============================================================================
# 4. آموزش
# ============================================================================
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

print("Starting training...")
start_time = time.time()
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100, batch_size=2048,
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[early_stopping]
)
print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

model.save('best_cylinder_model.keras')