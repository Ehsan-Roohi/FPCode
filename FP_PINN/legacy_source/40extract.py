#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
فاز 1: اسکریپت استخراج پارامترها

این اسکریپت یک بار اجرا می‌شود تا:
1. مدل Keras (.keras) را بارگذاری کند.
2. اسکالرهای Scikit-learn (.pkl) را بارگذاری کند.
3. تمام وزن‌ها، بایاس‌ها، میانگین‌ها و مقیاس‌ها را استخراج کند.
4. همه‌ی این پارامترها را در یک فایل واحد .npz ذخیره کند
   که مستقیماً توسط CuPy قابل خواندن است.
"""

import numpy as np
import joblib
import tensorflow as tf
import os

# --- نام فایل‌های ورودی ---
MODEL_FILE = 'fp_model.keras'
SCALER_X_FILE = 'scaler_X.pkl'
SCALER_Y_FILE = 'scaler_y.pkl'

# --- نام فایل خروجی ---
NPZ_FILE = 'model_params_for_cupy.npz'

def main():
    print(f"در حال بارگذاری مدل از {MODEL_FILE}...")
    # ابتدا مطمئن شوید که TensorFlow حافظه GPU را اشغال نکند
    tf.config.set_visible_devices([], 'GPU')
    
    model = tf.keras.models.load_model(MODEL_FILE)
    print("مدل بارگذاری شد.")
    
    print(f"در حال بارگذاری اسکالرها از {SCALER_X_FILE} و {SCALER_Y_FILE}...")
    try:
        scaler_X = joblib.load(SCALER_X_FILE)
        scaler_y = joblib.load(SCALER_Y_FILE)
        print("اسکالرها بارگذاری شدند.")
    except FileNotFoundError as e:
        print(f"خطا: فایل اسکالر پیدا نشد: {e}")
        return
    except Exception as e:
        print(f"خطا در بارگذاری اسکالر: {e}")
        return

    # دیکشنری برای ذخیره تمام پارامترها
    params_to_save = {}

    # 1. استخراج پارامترهای Scaler
    params_to_save['X_mean'] = scaler_X.mean_
    params_to_save['X_scale'] = scaler_X.scale_
    params_to_save['y_mean'] = scaler_y.mean_
    params_to_save['y_scale'] = scaler_y.scale_
    
    print("پارامترهای اسکالر استخراج شدند.")

    # 2. استخراج پارامترهای مدل (وزن‌ها و بایاس‌ها)
    # بر اساس خلاصه مدلی که قبلاً دیدیم:
    # لایه 0: dense (256)
    # لایه 1: dropout
    # لایه 2: dense_1 (256)
    # لایه 3: dropout_1
    # لایه 4: dense_2 (128)
    # لایه 5: dense_3 (9)
    try:
        weights_0 = model.layers[0].get_weights()
        params_to_save['W1'] = weights_0[0]  # وزن‌های لایه 1
        params_to_save['b1'] = weights_0[1]  # بایاس لایه 1

        weights_2 = model.layers[2].get_weights()
        params_to_save['W2'] = weights_2[0]  # وزن‌های لایه 2
        params_to_save['b2'] = weights_2[1]  # بایاس لایه 2

        weights_4 = model.layers[4].get_weights()
        params_to_save['W3'] = weights_4[0]  # وزن‌های لایه 3
        params_to_save['b3'] = weights_4[1]  # بایاس لایه 3

        weights_5 = model.layers[5].get_weights()
        params_to_save['W4'] = weights_5[0]  # وزن‌های لایه 4 (خروجی)
        params_to_save['b4'] = weights_5[1]  # بایاس لایه 4 (خروجی)
        
        print("وزن‌ها و بایاس‌های مدل استخراج شدند.")

    except Exception as e:
        print(f"خطا در استخراج وزن‌های مدل: {e}")
        print("آیا معماری مدل با چیزی که انتظار داشتیم متفاوت است؟")
        model.summary()
        return

    # 3. ذخیره همه چیز در فایل .npz
    try:
        np.savez(NPZ_FILE, **params_to_save)
        print("\n" + "="*50)
        print(f"موفقیت! پارامترها در '{NPZ_FILE}' ذخیره شدند.")
        print("="*50)
    except Exception as e:
        print(f"خطا در ذخیره فایل .npz: {e}")

if __name__ == "__main__":
    main()