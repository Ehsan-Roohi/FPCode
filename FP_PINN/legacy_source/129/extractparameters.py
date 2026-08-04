import numpy as np
from tensorflow import keras
import joblib

# ۱. بارگذاری مدل و اسکیلرها
model = keras.models.load_model('best_cylinder_model.keras')
X_scaler = joblib.load('X_scaler_cylinder.joblib')
y_scaler = joblib.load('y_scaler_cylinder.joblib')

# ۲. استخراج وزن‌ها و بایاس‌ها برای ۴ لایه مخفی و ۱ لایه خروجی
weights = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, keras.layers.Dense):
        w, b = layer.get_weights()
        weights[f'W{i+1}'] = w
        weights[f'b{i+1}'] = b

# ۳. استخراج پارامترهای اسکیلر (Mean و Scale)
weights['X_mean'] = X_scaler.mean_
weights['X_scale'] = X_scaler.scale_
weights['y_mean'] = y_scaler.mean_
weights['y_scale'] = y_scaler.scale_

# ۴. ذخیره نهایی در قالب NPZ
np.savez('native_cylinder_model_params.npz', **weights)
print("Parameters extracted to 'native_cylinder_model_params.npz'. Ready for solver integration.")