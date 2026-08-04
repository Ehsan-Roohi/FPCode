#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
*** ML Model Training Script (COMPARISON STUDY) ***

This script trains three different network architectures:
1. Small (Underfitting test)
2. Baseline (Current Proposed Model)
3. Large (Overfitting/Capacity test)

It generates the comparison metrics required by the reviewer.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import pandas as pd # For nice table formatting (optional, using print if not available)

print(f"Using TensorFlow version: {tf.__version__}")

# --- GPU Memory Growth Setup ---
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

# ============================================================================
# 1. Load Data
# ============================================================================
print("Loading training data from 'physics_training_data_5kn.npz'...")
try:
    data = np.load('physics_training_data_5kn.npz') 
    X = data['X'] 
    y = data['y'] 
    print(f"Data loaded successfully. Shapes: X={X.shape}, y={y.shape}")
except FileNotFoundError:
    print("Error: 'physics_training_data_5kn.npz' not found.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# ============================================================================
# 2. Data Splitting
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
# 3. Data Standardization
# ============================================================================
print("Standardizing data...")

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

# Save scalers once (same for all models)
joblib.dump(X_scaler, 'X_scaler.joblib')
joblib.dump(y_scaler, 'y_scaler.joblib')
print("Scalers saved.")

# ============================================================================
# 4. Model Definitions & Training Loop
# ============================================================================

def build_model(arch_type):
    """Creates a Keras model based on the requested architecture type."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(16,), name='input_features'))
    
    if arch_type == 'small':
        # Small: 2 layers of 128 neurons
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        
    elif arch_type == 'baseline':
        # Baseline (Proposed): 4 layers of 256 neurons
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        
    elif arch_type == 'large':
        # Large: 4 layers of 512 neurons
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        
    model.add(layers.Dense(9, activation='linear', name='output_coefficients'))
    return model

# Architectures to test
architectures = ['small', 'baseline', 'large']
results = []

print("\n" + "="*50)
print("STARTING COMPARISON TRAINING LOOP")
print("="*50)

for arch in architectures:
    print(f"\n---> Training Architecture: {arch.upper()}")
    
    # Clear session to free up GPU memory from previous model
    tf.keras.backend.clear_session()
    
    model = build_model(arch)
    
    # Count parameters
    param_count = model.count_params()
    print(f"     Parameter count: {param_count:,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error', 
        metrics=['mean_absolute_error']
    )
    
    # Callbacks
    model_name = f'best_model_{arch}.keras'
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            verbose=1, 
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            model_name, 
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # Train
    start_time = time.time()
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=100, # Max epochs
        batch_size=1024, 
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=callbacks,
        verbose=2 # Less clutter
    )
    end_time = time.time()
    
    # Record Metrics
    final_val_loss = min(history.history['val_loss'])
    training_time = (end_time - start_time) / 60.0 # Minutes
    epochs_run = len(history.history['loss'])
    
    results.append({
        'Architecture': arch,
        'Params': param_count,
        'Val_Loss_MSE': final_val_loss,
        'Time_mins': training_time,
        'Epochs': epochs_run
    })
    
    print(f"     Finished {arch}. Best Val Loss: {final_val_loss:.6f}")

# ============================================================================
# 5. Final Report
# ============================================================================
print("\n" + "="*60)
print("FINAL COMPARISON RESULTS (For Reviewer Response)")
print("="*60)
print(f"{'Architecture':<15} | {'Params':<10} | {'Val MSE Loss':<15} | {'Time (min)':<10}")
print("-" * 60)

for r in results:
    print(f"{r['Architecture']:<15} | {r['Params']:<10,} | {r['Val_Loss_MSE']:<15.6e} | {r['Time_mins']:<10.2f}")

print("-" * 60)
print("NOTE: Copy these values into your response letter table.")