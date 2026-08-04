#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
*** ML Model Training Script (DEEPER NETWORK) ***

This script loads the generated data (physics_training_data_5kn.npz),
prepares it, builds a DEEPER and WIDER neural network, trains it,
and saves the final model and scalers.
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

# --- تنظیم رشد حافظه TensorFlow (برای جلوگیری از خطا) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Set TensorFlow memory growth to True for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU found for TensorFlow.")
# --- پایان بخش جدید ---

# ============================================================================
# 1. بارگذاری داده‌ها (Loading Data)
# ============================================================================
print("Loading training data from 'physics_training_data_5kn.npz'...")
try:
    # از فایل داده‌های تمیز (که با جاب طولانی ساختید) استفاده کنید
    data = np.load('physics_training_data_5kn.npz') 
    X = data['X'] 
    y = data['y'] 
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
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")

del X
del y

# ============================================================================
# 3. استانداردسازی داده‌ها (Data Standardization)
# ============================================================================
print("Standardizing data... (Fitting scalers on training data ONLY)")

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

print("Data scaled successfully.")

# --- ذخیره اسکیلرها ---
joblib.dump(X_scaler, 'X_scaler.joblib')
joblib.dump(y_scaler, 'y_scaler.joblib')
print("Scalers 'X_scaler.joblib' and 'y_scaler.joblib' saved.")

# ============================================================================
# 4. ساخت مدل شبکه عصبی (*** مدل عمیق‌تر ***)
# ============================================================================
print("Building the DEEPER neural network model (4x256)...")

# 16 inputs -> 256 -> 256 -> 256 -> 256 -> 9 outputs
model = keras.Sequential([
    layers.Input(shape=(16,), name='input_features'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'), # <-- لایه چهارم (عمیق‌تر)
    layers.Dense(9, activation='linear', name='output_coefficients')
])

model.summary()

# ============================================================================
# 5. کامپایل و آموزش مدل (Compile and Train Model)
# ============================================================================
print("Compiling the model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), # نرخ یادگیری پایین برای تنظیم دقیق
    loss='mean_squared_error', 
    metrics=['mean_absolute_error']
)

# --- Callbacks ---
# 1. EarlyStopping: توقف پس از 10 اپاک عدم بهبود
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, # صبوری بیشتر برای مدل بزرگتر
    verbose=1, 
    restore_best_weights=True
)

# 2. ModelCheckpoint: ذخیره بهترین مدل
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras', 
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("Starting model training (this may take longer)...")
start_time = time.time()

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100, # اجازه می‌دهیم بیشتر اجرا شود
    batch_size=1024, 
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[early_stopping, model_checkpoint]
)

end_time = time.time()
print(f"Training finished in {(end_time - start_time) / 60.0:.2f} minutes.")
print("The best model has been saved as 'best_model.keras'.")