#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLR (Çoklu Doğrusal Regresyon) modelini oluşturma ve eğitme işlemlerini içeren modül
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mlr_modelini_olustur():
    """
    Çoklu Doğrusal Regresyon modelini oluşturur
    
    Döndürülen:
    Model: Oluşturulan MLR modeli
    """
    print("Çoklu Doğrusal Regresyon modeli oluşturuluyor...")
    
    # MLR modeli
    model = LinearRegression()
    
    return model

def mlr_modelini_egit(model, X_train, y_train):
    """
    Çoklu Doğrusal Regresyon modelini eğitir
    
    Parametreler:
    model (Model): Eğitilecek MLR modeli
    X_train (DataFrame): Eğitim verisi - Bağımsız değişkenler
    y_train (Series): Eğitim verisi - Hedef değişken
    
    Döndürülen:
    Model: Eğitilmiş MLR modeli
    """
    print("Çoklu Doğrusal Regresyon modeli eğitiliyor...")
    
    # Modeli eğitme
    model.fit(X_train, y_train)
    
    # Katsayıları ve sabit terimi yazdırma
    print("Model katsayıları:")
    for i, feature in enumerate(X_train.columns):
        print(f"- {feature}: {model.coef_[i]:.4f}")
    print(f"- Sabit terim (intercept): {model.intercept_:.4f}")
    
    # R2 skorunu yazdırma
    train_r2 = model.score(X_train, y_train)
    print(f"Eğitim verisi R² skoru: {train_r2:.4f}")
    
    return model

def mlr_tahmin_et(model, X_test):
    """
    MLR modeli ile tahmin yapar
    
    Parametreler:
    model (Model): Eğitilmiş MLR modeli
    X_test (DataFrame): Test verisi - Bağımsız değişkenler
    
    Döndürülen:
    array: Tahmin edilen değerler
    """
    print("MLR modeli ile tahmin yapılıyor...")
    y_pred = model.predict(X_test)
    
    return y_pred

def mlr_degerlendirme(y_test, y_pred):
    """
    MLR modelinin performansını değerlendirir
    
    Parametreler:
    y_test (Series): Gerçek test değerleri
    y_pred (array): Tahmin edilen değerler
    
    Döndürülen:
    dict: Performans metrikleri
    """
    print("MLR modeli performansı değerlendiriliyor...")
    
    # Temel metrikler hesaplama
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Sonuçları yazdırma
    print(f"MLR Modeli Değerlendirme Sonuçları:")
    print(f"- MSE (Ortalama Kare Hata): {mse:.4f}")
    print(f"- RMSE (Kök Ortalama Kare Hata): {rmse:.4f}")
    print(f"- MAE (Ortalama Mutlak Hata): {mae:.4f}")
    print(f"- R² Skoru: {r2:.4f}")
    
    # Görselleştirme - Gerçek vs Tahmin grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.title('MLR Modeli: Gerçek vs Tahmin')
    plt.tight_layout()
    plt.savefig('results/mlr_predictions.png')
    print("Tahmin sonuçları görselleştirildi: results/mlr_predictions.png")
    
    # Metrikleri bir sözlükte saklama
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics

def mlr_modelini_kaydet(model, model_path="models/mlr_student_performance_model.pkl"):
    """
    Eğitilmiş MLR modelini dosyaya kaydeder
    
    Parametreler:
    model (Model): Kaydedilecek model
    model_path (str): Model dosyasının yolu
    """
    print(f"MLR modeli kaydediliyor: {model_path}...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model başarıyla kaydedildi.")

def mlr_modelini_yukle(model_path="models/mlr_student_performance_model.pkl"):
    """
    Kaydedilmiş MLR modelini yükler
    
    Parametreler:
    model_path (str): Model dosyasının yolu
    
    Döndürülen:
    Model: Yüklenen MLR modeli
    """
    print(f"MLR modeli yükleniyor: {model_path}...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model başarıyla yüklendi.")
    return model

def ozellik_onem_analizi(model, X_test):
    """
    MLR modeli için özelliklerin önemini analiz eder
    
    Parametreler:
    model (Model): Eğitilmiş MLR modeli
    X_test (DataFrame): Test veri seti
    
    Döndürülen:
    DataFrame: Özelliklerin önem skorları
    """
    print("MLR modeli özellik önem analizi yapılıyor...")
    
    # Mutlak katsayı değerlerini hesaplama
    feature_importance = np.abs(model.coef_)
    
    # Sonuçları DataFrame'e çevirme
    importance_df = pd.DataFrame({
        'Özellik': X_test.columns,
        'Önem_Skoru': feature_importance
    })
    
    # Önem skoruna göre sıralama
    importance_df = importance_df.sort_values('Önem_Skoru', ascending=False)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Özellik'], importance_df['Önem_Skoru'])
    plt.xlabel('Mutlak Katsayı Değeri')
    plt.ylabel('Özellik')
    plt.title('MLR Modeli - Özellik Önem Analizi')
    plt.tight_layout()
    plt.savefig('results/mlr_feature_importance.png')
    print("Özellik önem analizi görselleştirildi: results/mlr_feature_importance.png")
    
    return importance_df

if __name__ == "__main__":
    print("Bu bir modül olarak tasarlanmıştır. Lütfen main.py üzerinden çalıştırın.")