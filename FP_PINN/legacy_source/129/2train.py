import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import matplotlib.pyplot as plt

# تنظیم اندازه فونت برای تمامی نمودارهای احتمالی طبق دستور شما
plt.rcParams.update({'font.size': 22})

print(f"Using TensorFlow version: {tf.__version__}")

# تنظیم رشد حافظه GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ============================================================================
# 1. بارگذاری و تصفیه داده‌ها (تغییر سوم: حذف Outliers)
# ============================================================================
print("Loading massive cylinder data...")
data = np.load('cylinder_training_data.npz')
X_raw = data['X']
y_raw = data['y']

print(f"Original samples: {X_raw.shape[0]}")

# حذف داده‌های پرت (Outliers): مقادیری که به دلیل نویز عددی در سلول‌های کم‌ذره منفجر شده‌اند
# ما فقط داده‌هایی را نگه می‌داریم که ضرایب آن‌ها در محدوده فیزیکی معقول باشد
mask = np.all(np.abs(y_raw) < 1e6, axis=1) 
X_filtered = X_raw[mask]
y_filtered = y_raw[mask]

print(f"Samples after filtering outliers: {X_filtered.shape[0]}")

# انتخاب تصادفی ۵ میلیون نمونه برای مدیریت RAM و سرعت آموزش
target_samples = 5000000 
if X_filtered.shape[0] > target_samples:
    indices = np.random.choice(X_filtered.shape[0], target_samples, replace=False)
    X = X_filtered[indices]
    y = y_filtered[indices]
else:
    X, y = X_filtered, y_filtered

del X_raw, y_raw, X_filtered, y_filtered # پاکسازی حافظه RAM

# ============================================================================
# 2. آماده‌سازی و اسکیل کردن (Standardization)
# ============================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

# ذخیره اسکیلرها برای استفاده در کد اصلی (Inference)
joblib.dump(X_scaler, 'X_scaler_cylinder.joblib')
joblib.dump(y_scaler, 'y_scaler_cylinder.joblib')
print("Scalers saved successfully.")

# ============================================================================
# 3. ساخت مدل عمیق (10 خروجی مخصوص سیلندر)
# ============================================================================
# ورودی شامل ۱۶ ویژگی فیزیکی و خروجی شامل ۱۰ ضریب Cubic FP است
model = keras.Sequential([
    layers.Input(shape=(16,), name='input_features'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='linear', name='output_coefficients')
])

# تغییر دوم: استفاده از Huber Loss برای پایداری در برابر گرادیان‌های شدید شوک
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-4), 
    loss=tf.keras.losses.Huber(), 
    metrics=['mae']
)

# ============================================================================
# 4. تنظیم Callbacks (تغییر اول: ReduceLROnPlateau)
# ============================================================================
# کاهش خودکار نرخ یادگیری در صورت عدم پیشرفت لاس
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-7, 
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_cylinder_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)

# ============================================================================
# 5. شروع آموزش (Training)
# ============================================================================
print("Starting training on A100 GPU...")
start_time = time.time()

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=8192, # افزایش سایز بچ برای استفاده حداکثری از توان A100
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

end_time = time.time()
print(f"Training finished in {(end_time - start_time) / 60.0:.2f} minutes.")

# ============================================================================
# 6. رسم نمودار لاس (با رعایت فونت ۲۲)
# ============================================================================
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Train Loss (Huber)')
plt.plot(history.history['val_loss'], label='Val Loss (Huber)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cylinder ML Model Training History')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
print("Loss plot saved as 'training_loss.png'.")