import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib

# تنظیمات
EPOCHS = 50
BATCH_SIZE = 1024
FILENAME_DATA = "training_data.npz"
FILENAME_PARAMS = "model_params.npz"

def train_and_export():
    print("Loading data...")
    try:
        data = np.load(FILENAME_DATA)
        X = data['inputs']  # 16 Features
        y = data['outputs'] # 9 Coefficients
    except FileNotFoundError:
        print(f"Error: {FILENAME_DATA} not found. Run the simulation in 'DATA_GEN' mode first.")
        return

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # نرمال‌سازی داده‌ها (استانداردسازی)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # معماری شبکه دقیقاً طبق مقاله (16 -> 256x4 -> 9)
    model = keras.Sequential([
        layers.Input(shape=(16,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(9, activation='linear') # Regression output
    ])

    model.compile(optimizer='adam', loss='mse')
    
    print("Starting training...")
    history = model.fit(X_scaled, y_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    print("Extracting weights for GPU-Native Inference...")
    weights = []
    for layer in model.layers:
        w, b = layer.get_weights()
        weights.append(w)
        weights.append(b)
    
    # ذخیره پارامترها به صورت آرایه NumPy برای لود کردن در Numba
    np.savez(FILENAME_PARAMS,
             mean_in=scaler_X.mean_, scale_in=scaler_X.scale_,
             mean_out=scaler_y.mean_, scale_out=scaler_y.scale_,
             W1=weights[0], b1=weights[1],
             W2=weights[2], b2=weights[3],
             W3=weights[4], b3=weights[5],
             W4=weights[6], b4=weights[7],
             W5=weights[8], b5=weights[9])
    
    print(f"Model parameters saved to {FILENAME_PARAMS}. Now you can run the simulation in 'INFERENCE' mode.")

if __name__ == "__main__":
    train_and_export()