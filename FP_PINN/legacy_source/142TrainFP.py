import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os

# --- ENVIRONMENT SETTINGS ---
# Disable XLA to reduce memory overhead in constrained environments
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        print(e)

# --- SETTINGS ---
EPOCHS = 50
# CRITICAL CHANGE: Reduced from 65536 to 1024 to fit in 500MB VRAM
BATCH_SIZE = 1024  
FILENAME_DATA = "training_data.npz"
FILENAME_PARAMS = "model_params.npz"
MAX_SAMPLES = 20_000_000 

def train_and_export():
    print("Loading data...")
    if not os.path.exists(FILENAME_DATA):
        print(f"Error: {FILENAME_DATA} not found.")
        return

    # Load data using mmap to save RAM
    data = np.load(FILENAME_DATA, mmap_mode='r') 
    X_full = data['inputs']
    y_full = data['outputs']
    
    total_samples = X_full.shape[0]
    print(f"Total dataset size: {total_samples}")

    # --- Subsampling Strategy ---
    if total_samples > MAX_SAMPLES:
        print(f"Dataset too large. Subsampling to {MAX_SAMPLES} random samples...")
        indices = np.random.choice(total_samples, MAX_SAMPLES, replace=False)
        X = X_full[indices]
        y = y_full[indices]
    else:
        X = np.array(X_full)
        y = np.array(y_full)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Shuffle on CPU
    X, y = shuffle(X, y, random_state=42)

    print("Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    print("Creating tf.data Pipeline (CPU -> GPU Streaming)...")
    
    # Force dataset creation on CPU to avoid VRAM OOM
    with tf.device('/CPU:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, y_scaled))
        
    # Shuffle and batch on CPU, then prefetch to GPU
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000) # Reduced buffer size to save RAM
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Define Model
    model = keras.Sequential([
        layers.Input(shape=(16,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(9, activation='linear') 
    ])

    # Standard Adam optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse')
    
    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=1)
    ]

    print(f"Starting training pipeline with Batch Size {BATCH_SIZE}...")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS, 
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print("\n!!! GPU Training Failed. If this is OOM, try setting CUDA_VISIBLE_DEVICES=-1 to force CPU mode !!!")
        raise e

    print("Extracting weights for GPU-Native Inference...")
    weights = []
    for layer in model.layers:
        w, b = layer.get_weights()
        weights.append(w)
        weights.append(b)
    
    np.savez(FILENAME_PARAMS,
             mean_in=scaler_X.mean_, scale_in=scaler_X.scale_,
             mean_out=scaler_y.mean_, scale_out=scaler_y.scale_,
             W1=weights[0], b1=weights[1],
             W2=weights[2], b2=weights[3],
             W3=weights[4], b3=weights[5],
             W4=weights[6], b4=weights[7],
             W5=weights[8], b5=weights[9])
    
    print(f"Model parameters saved to {FILENAME_PARAMS}.")

if __name__ == "__main__":
    train_and_export()