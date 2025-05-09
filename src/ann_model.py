def ann_degerlendirme(y_test, y_pred):
    """
    YSA modelinin performansını değerlendirir
    
    Parametreler:
    y_test (Series): Gerçek test değerleri
    y_pred (array): Tahmin edilen değerler
    
    Döndürülen:
    dict: Performans metrikleri
    """
    print("YSA modeli performansı değerlendiriliyor...")
    
    # Temel metrikler hesaplama
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Sonuçları yazdırma
    print(f"YSA Modeli Değerlendirme Sonuçları:")
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
    plt.title('YSA Modeli: Gerçek vs Tahmin')
    plt.tight_layout()
    plt.savefig('results/ann_predictions.png')
    print("Tahmin sonuçları görselleştirildi: results/ann_predictions.png")
    
    # Metrikleri bir sözlükte saklama
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics

def ann_modelini_kaydet(model, model_path="models/ann_student_performance_model.h5"):
    """
    Eğitilmiş YSA modelini dosyaya kaydeder
    
    Parametreler:
    model (Model): Kaydedilecek model
    model_path (str): Model dosyasının yolu
    """
    print(f"YSA modeli kaydediliyor: {model_path}...")
    model.save(model_path)
    print("Model başarıyla kaydedildi.")

def ann_modelini_yukle(model_path="models/ann_student_performance_model.h5"):
    """
    Kaydedilmiş YSA modelini yükler
    
    Parametreler:
    model_path (str): Model dosyasının yolu
    
    Döndürülen:
    Model: Yüklenen YSA modeli
    """
    print(f"YSA modeli yükleniyor: {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model başarıyla yüklendi.")
    return model

def ozellik_onem_analizi(model, X_test, feature_names=None):
    """
    Özelliklerin önemini analiz eder (Permutasyon önem analizi)
    
    Parametreler:
    model (Model): Eğitilmiş YSA modeli
    X_test (DataFrame): Test veri seti
    feature_names (list): Özellik isimleri listesi (sağlanmazsa X_test sütun isimleri kullanılır)
    
    Döndürülen:
    DataFrame: Özelliklerin önem skorları
    """
    print("Özellik önem analizi yapılıyor...")
    
    if feature_names is None and hasattr(X_test, 'columns'):
        feature_names = X_test.columns
    
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    # Baseline performans
    baseline_pred = model.predict(X_test_np).flatten()
    baseline_mse = mean_squared_error(baseline_pred, baseline_pred)  # Her zaman 0 olacak
    
    importance_scores = []
    
    for i in range(X_test_np.shape[1]):
        # Orijinal veri kopyası
        X_permuted = X_test_np.copy()
        # i. sütundaki verileri karıştırma
        np.random.shuffle(X_permuted[:, i])
        # Karıştırılmış veri ile tahmin
        permuted_pred = model.predict(X_permuted).flatten()
        # Performans düşüşünü hesaplama
        permuted_mse = mean_squared_error(baseline_pred, permuted_pred)
        # Önem skoru (performans düşüşü)
        importance = permuted_mse - baseline_mse
        importance_scores.append(importance)
    
    # Sonuçları DataFrame'e çevirme
    importance_df = pd.DataFrame({
        'Özellik': feature_names,
        'Önem_Skoru': importance_scores
    })
    
    # Önem skoruna göre sıralama
    importance_df = importance_df.sort_values('Önem_Skoru', ascending=False)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Özellik'], importance_df['Önem_Skoru'])
    plt.xlabel('Önem Skoru (MSE Artışı)')
    plt.ylabel('Özellik')
    plt.title('YSA Modeli - Özellik Önem Analizi')
    plt.tight_layout()
    plt.savefig('results/ann_feature_importance.png')
    print("Özellik önem analizi görselleştirildi: results/ann_feature_importance.png")
    
    return importance_df

if __name__ == "__main__":
    print("Bu bir modül olarak tasarlanmıştır. Lütfen main.py üzerinden çalıştırın.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def ann_modelini_olustur(input_dim):
    """
    YSA modelini oluşturur
    """
    print("YSA modeli oluşturuluyor...")
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Model başarıyla oluşturuldu.")
    return model

def ann_modelini_egit(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    """
    YSA modelini eğitir
    """
    print("YSA modeli eğitiliyor...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    print("Model eğitimi tamamlandı.")
    return history, model

def ann_tahmin_et(model, X_test):
    """
    YSA modeli ile tahmin yapar
    """
    print("YSA modeli ile tahmin yapılıyor...")
    y_pred = model.predict(X_test).flatten()
    return y_pred