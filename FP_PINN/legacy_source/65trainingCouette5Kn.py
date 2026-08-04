#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
*** ML Model Training Script ***

This script loads the generated data (physics_training_data_5kn.npz),
prepares it, builds a neural network, trains it, and saves the
final model and the crucial data scalers for future use.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # For saving scalers
import time

print(f"Using TensorFlow version: {tf.__version__}")

# ============================================================================
# 1. بارگذاری داده‌ها (Loading Data)
# ============================================================================
print("Loading training data from 'physics_training_data_5kn.npz'...")
try:
    data = np.load('physics_training_data_5kn.npz')
    X = data['X'] # Input features (4500000, 16)
    y = data['y'] # Output labels (4500000, 9)
    print(f"Data loaded successfully. Shapes: X={X.shape}, y={y.shape}")
except FileNotFoundError:
    print("Error: 'physics_training_data_5kn.npz' not found.")
    print("Please ensure the data generation script ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# ============================================================================
# 2. جداسازی داده‌ها (Train/Validation Split)
# ============================================================================
print("Splitting data into training and validation sets (90% train, 10% val)...")
# Using 10% for validation (450,000 samples) is more than enough
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")

# Clear original large arrays from memory
del X
del y

# ============================================================================
# 3. استانداردسازی داده‌ها (Data Standardization) - بسیار مهم!
# ============================================================================
print("Standardizing data... (Fitting scalers on training data ONLY)")
# ورودی‌ها (X) و خروجی‌ها (y) واحدهای فیزیکی کاملاً متفاوتی دارند
# (چگالی، دما، فشار، ...). شبکه عصبی تنها زمانی به خوبی کار می‌کند
# که همه ورودی‌ها و خروجی‌ها در یک مقیاس مشابه (نزدیک به 0) باشند.

# Fit the scaler ONLY on the training data
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)

# Use the SAME scaler to transform the validation data
X_val_scaled = X_scaler.transform(X_val)

# Repeat for the output labels (y)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

print("Data scaled successfully.")

# --- ذخیره اسکیلرها (Saving Scalers) ---
# ما به این اسکیلرها در زمان تست نیاز داریم تا ورودی خام را به
# ورودی مقیاس‌شده تبدیل کنیم و پیش‌بینی مقیاس‌شده را به
# ضرایب فیزیکی واقعی برگردانیم.
joblib.dump(X_scaler, 'X_scaler.joblib')
joblib.dump(y_scaler, 'y_scaler.joblib')
print("Scalers 'X_scaler.joblib' and 'y_scaler.joblib' saved.")

# ============================================================================
# 4. ساخت مدل شبکه عصبی (Build Model Architecture)
# ============================================================================
print("Building the neural network model...")

# A simple but robust Multi-Layer Perceptron (MLP)
# 16 inputs -> 128 -> 128 -> 128 -> 9 outputs
model = keras.Sequential([
    layers.Input(shape=(16,), name='input_features'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='linear', name='output_coefficients') # 'linear' for regression
])

model.summary()

# ============================================================================
# 5. کامپایل و آموزش مدل (Compile and Train Model)
# ============================================================================
print("Compiling the model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='mean_squared_error', # Best loss for regression
    metrics=['mean_absolute_error'] # More interpretable metric
)

# --- Callbacks ---
# 1. EarlyStopping: اگر مدل پس از 5 اپاک بهبود پیدا نکرد، آموزش را متوقف کن
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    verbose=1, 
    restore_best_weights=True
)

# 2. ModelCheckpoint: فقط بهترین مدل را (بر اساس کمترین val_loss) ذخیره کن
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras', # Using the modern .keras format
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("Starting model training...")
start_time = time.time()

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50, # A good starting point
    batch_size=1024, # Large batch size for this large dataset
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[early_stopping, model_checkpoint]
)

end_time = time.time()
print(f"Training finished in {(end_time - start_time) / 60.0:.2f} minutes.")
print("The best model has been saved as 'best_model.keras'.")