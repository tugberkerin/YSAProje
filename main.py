#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Öğrenci Performans Tahmin Modeli - Ana Çalıştırma Dosyası
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Proje modüllerini içe aktarma
from src.ann_model import (
    ann_modelini_olustur, ann_modelini_egit, ann_tahmin_et, 
    ann_degerlendirme, ann_modelini_kaydet, ozellik_onem_analizi as ann_ozellik_onem_analizi
)
from mlr_model import (
    mlr_modelini_olustur, mlr_modelini_egit, mlr_tahmin_et, 
    mlr_degerlendirme, mlr_modelini_kaydet, ozellik_onem_analizi as mlr_ozellik_onem_analizi
)
from model_degerlendirme import (
    model_performanslari_karsilastir, feature_importance_karsilastir, en_etkili_ozellikleri_belirle
)

def veri_hazirla(veri_dosyasi, hedef_degisken='final_grade'):
    """
    Veriyi okur ve işlem için hazırlar
    
    Parametreler:
    veri_dosyasi (str): Veri dosyasının yolu
    hedef_degisken (str): Hedef değişkenin adı
    
    Döndürülen:
    tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print(f"Veri hazırlanıyor: {veri_dosyasi}...")
    
    # Veriyi okuma
    df = pd.read_csv(veri_dosyasi)
    print(f"Veri boyutu: {df.shape}")
    
    # Temel bilgileri gösterme
    print("\nVeri setinin ilk 5 satırı:")
    print(df.head())
    
    print("\nVeri seti istatistikleri:")
    print(df.describe())
    
    # Kategorik değişkenleri one-hot encoding yapma
    categorical_columns = df.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        print(f"\nKategorik değişkenler one-hot encoding yapılıyor: {list(categorical_columns)}")
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Hedef değişken ve öznitelikleri ayırma
    if hedef_degisken in df.columns:
        X = df.drop(hedef_degisken, axis=1)
        y = df[hedef_degisken]
    else:
        raise ValueError(f"Hedef değişken '{hedef_degisken}' veri setinde bulunamadı!")
    
    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # Öznitelikleri ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ölçeklendirilmiş verileri DataFrame'e dönüştürme
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Ölçeklendirici nesnesini kaydetme
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Ölçeklendirici nesnesi kaydedildi: models/scaler.pkl")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler

def main():
    """
    Ana fonksiyon
    """
    print("Öğrenci Performans Tahmin Modeli uygulaması başlatılıyor...")
    
    # Gerekli klasörleri oluşturma
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Veriyi hazırlama - Verinizin yolunu burada belirtin
    veri_dosyasi = 'student_performance.csv'  # Bu dosyayı oluşturmanız veya değiştirmeniz gerekebilir
    X_train, X_test, y_train, y_test, scaler = veri_hazirla(veri_dosyasi)
    
    # ============================
    # MLR Modeli
    # ============================
    print("\n" + "="*50)
    print("MLR (Çoklu Doğrusal Regresyon) Modeli")
    print("="*50)
    
    # MLR modelini oluşturma ve eğitme
    mlr_model = mlr_modelini_olustur()
    mlr_model = mlr_modelini_egit(mlr_model, X_train, y_train)
    
    # MLR ile tahmin yapma
    mlr_predictions = mlr_tahmin_et(mlr_model, X_test)
    
    # MLR performans değerlendirmesi
    mlr_metrics = mlr_degerlendirme(y_test, mlr_predictions)
    
    # MLR özellik önem analizi
    mlr_importance = mlr_ozellik_onem_analizi(mlr_model, X_test)
    
    # MLR modelini kaydetme
    mlr_modelini_kaydet(mlr_model)
    
    # ============================
    # ANN Modeli
    # ============================
    print("\n" + "="*50)
    print("ANN (Yapay Sinir Ağı) Modeli")
    print("="*50)
    
    # ANN modelini oluşturma
    ann_model = ann_modelini_olustur(X_train.shape[1])
    
    # ANN modelini eğitme
    history, ann_model = ann_modelini_egit(
        ann_model, X_train, y_train, 
        epochs=150, batch_size=32, validation_split=0.2
    )
    
    # ANN ile tahmin yapma
    ann_predictions = ann_tahmin_et(ann_model, X_test)
    
    # ANN performans değerlendirmesi
    ann_metrics = ann_degerlendirme(y_test, ann_predictions)
    
    # ANN özellik önem analizi
    ann_importance = ann_ozellik_onem_analizi(ann_model, X_test, X_test.columns)
    
    # ANN modelini kaydetme
    ann_modelini_kaydet(ann_model)
    
    # ============================
    # Model Karşılaştırması
    # ============================
    print("\n" + "="*50)
    print("Model Karşılaştırması")
    print("="*50)
    
    # Model performanslarını karşılaştırma
    model_metrics = {
        'MLR': mlr_metrics,
        'ANN': ann_metrics
    }
    comparison_df, best_models = model_performanslari_karsilastir(model_metrics)
    
    # Özellik önem analizlerini karşılaştırma
    combined_importance = feature_importance_karsilastir(mlr_importance, ann_importance)
    
    # En etkili özellikleri belirleme
    top_features = en_etkili_ozellikleri_belirle(combined_importance, top_n=5)
    
    print("\nUygulama başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
