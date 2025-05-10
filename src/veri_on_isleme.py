#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri ön işleme adımlarını içeren modül
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def veri_on_isleme(df):
    """
    Veri setini temizler, dönüştürür ve ön işleme adımlarını uygular
    
    Parametreler:
    df (DataFrame): Ham veri seti
    
    Döndürülen:
    tuple: (temizlenmiş_df, ölçeklendirilmiş_df, scaler)
    """
    print("Veri ön işleme adımları başlatılıyor...")
    
    # Veri setinin kopyasını oluşturma
    df_cleaned = df.copy()
    
    # 1. Gereksiz Sütunları Kaldırma
    if "Student_ID" in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=["Student_ID"])
        print("- 'Student_ID' sütunu kaldırıldı")
    
    if "Gender" in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=["Gender"])
        print("- 'Gender' sütunu kaldırıldı")
    
    if "Preferred_Learning_Style" in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=["Preferred_Learning_Style"])
        print("- 'Preferred_Learning_Style' sütunu kaldırıldı")
    
    # 2. Eksik Verileri Kontrol Etme ve Temizleme
    eksik_degerler = df_cleaned.isnull().sum()
    if eksik_degerler.sum() > 0:
        print("- Eksik değerler tespit edildi. Doldurma işlemi uygulanıyor...")
        
        # Sayısal sütunlar için ortalama ile doldurma
        sayisal_sutunlar = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for sutun in sayisal_sutunlar:
            if df_cleaned[sutun].isnull().sum() > 0:
                df_cleaned[sutun].fillna(df_cleaned[sutun].mean(), inplace=True)
                print(f"  - '{sutun}' sütunundaki eksik değerler ortalama ile dolduruldu")
        
        # Kategorik sütunlar için mod ile doldurma
        kategorik_sutunlar = df_cleaned.select_dtypes(include=['object']).columns
        for sutun in kategorik_sutunlar:
            if df_cleaned[sutun].isnull().sum() > 0:
                df_cleaned[sutun].fillna(df_cleaned[sutun].mode()[0], inplace=True)
                print(f"  - '{sutun}' sütunundaki eksik değerler mod ile dolduruldu")
    else:
        print("- Eksik değer bulunmamaktadır")
    
    # 3. Kategorik Verileri Sayısallaştırma
    # İkili kategorik değişkenler için
    ikili_kategorik = {
        "Participation_in_Discussions": {"Yes": 1, "No": 0},
        "Use_of_Educational_Tech": {"Yes": 1, "No": 0}
    }
    
    for sutun, mapping in ikili_kategorik.items():
        if sutun in df_cleaned.columns:
            df_cleaned[sutun] = df_cleaned[sutun].map(mapping)
            print(f"- '{sutun}' sütunu sayısallaştırıldı: {mapping}")
    
    # Çoklu kategorik değişkenler için one-hot encoding
    coklu_kategorik = ["Self_Reported_Stress_Level"]
    
    for sutun in coklu_kategorik:
        if sutun in df_cleaned.columns:
            df_cleaned = pd.get_dummies(df_cleaned, columns=[sutun], drop_first=True)
            print(f"- '{sutun}' sütunu için one-hot encoding uygulandı")
    
    # Hedef değişkeni belirleme
    hedef_degisken = None
    muhtemel_hedef_sutunlar = ["Exam_Score", "Final_Grade", "Sınav_Puanı"]
    
    for sutun in muhtemel_hedef_sutunlar:
        if sutun in df_cleaned.columns:
            hedef_degisken = sutun
            break
    
    if not hedef_degisken:
        raise ValueError("Hedef değişken bulunamadı! Veri setinizde sınav puanını içeren sütun ismi farklı olabilir.")
    
    print(f"- Hedef değişken: '{hedef_degisken}'")
    
    # 4. Sayısal Verileri Ölçeklendirme
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cleaned),
        columns=df_cleaned.columns
    )
    print("- Tüm sayısal değerler 0-1 aralığına ölçeklendirildi")
    
    # Korelasyon analizi ve görselleştirme
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_cleaned.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig("results/korelasyon_matrisi.png")
    print("- Korelasyon matrisi oluşturuldu ve kaydedildi: results/korelasyon_matrisi.png")
    
    # Hedef değişken ile en yüksek korelasyona sahip özelliklerin belirlenmesi
    korelasyonlar = correlation_matrix[hedef_degisken].sort_values(ascending=False)
    print(f"\nHedef değişken '{hedef_degisken}' ile en yüksek korelasyona sahip özellikler:")
    print(korelasyonlar)
    
    return df_cleaned, df_scaled, scaler

def veriyi_bol(df_scaled):
    """
    Veriyi eğitim ve test kümelerine ayırır
    
    Parametreler:
    df_scaled (DataFrame): Ölçeklendirilmiş veri seti
    
    Döndürülen:
    tuple: (X_train, X_test, y_train, y_test)
    """
    # Hedef değişkeni belirleme
    hedef_degisken = None
    muhtemel_hedef_sutunlar = ["Exam_Score", "Final_Grade", "Sınav_Puanı"]
    
    for sutun in muhtemel_hedef_sutunlar:
        if sutun in df_scaled.columns:
            hedef_degisken = sutun
            break
    
    if not hedef_degisken:
        raise ValueError("Hedef değişken bulunamadı! Veri setinizde sınav puanını içeren sütun ismi farklı olabilir.")
    
    # Bağımsız değişkenler (X) ve hedef değişken (y) ayırma
    X = df_scaled.drop(columns=[hedef_degisken])
    y = df_scaled[hedef_degisken]
    
    # Veriyi eğitim ve test olarak bölme (%80 eğitim, %20 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Veri bölündü:")
    print(f"- Eğitim seti: {X_train.shape[0]} örnek ({X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%})")
    print(f"- Test seti: {X_test.shape[0]} örnek ({X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%})")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Bu modül doğrudan çalıştırıldığında test amaçlı olarak
    import pandas as pd
    from veri_yukleme import veri_setini_yukle
    
    df = veri_setini_yukle("../data/student_performance_data.csv")
    if df is not None:
        df_cleaned, df_scaled, scaler = veri_on_isleme(df)
        X_train, X_test, y_train, y_test = veriyi_bol(df_scaled)