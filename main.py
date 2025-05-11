import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. CSV Dosyasını Yükleme
def load_data(file_path):
    """CSV dosyasını yükle."""
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # Eğer kodlama sorunu varsa farklı kodlamalarla deneyelim
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except:
            try:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
            except:
                # Excel dosyası olabilir
                df = pd.read_excel(file_path)
    
    print(f"Yüklenen veri boyutu: {df.shape}")
    return df

# 2. Veri Analizi ve Görselleştirme
def analyze_data(df):
    """Temel veri analizi ve görselleştirme."""
    print("İlk 5 satır:")
    print(df.head())
    
    print("\nVeri tipi bilgisi:")
    print(df.info())
    
    print("\nİstatistik özeti:")
    print(df.describe())
    
    print("\nEksik değerler:")
    print(df.isnull().sum())
    
    # Kategorik sütunları tespit et
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print("\nKategorik sütunlar:", categorical_cols.tolist())
    
    # Sayısal sütunları tespit et
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nSayısal sütunlar:", numerical_cols.tolist())
    
    # Korelasyon matrisi (sadece sayısal sütunlar için)
    plt.figure(figsize=(12, 10))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelasyon Matrisi')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Kategorik sütunlar için dağılım grafikleri
    for col in categorical_cols[:5]:  # İlk 5 kategorik sütun için
        plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'{col} Dağılımı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return corr

# 3. Gereksiz Sütunları Silme
def drop_columns(df, columns_to_drop):
    """Belirtilen sütunları sil."""
    # Sütunların var olup olmadığını kontrol et
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    if len(existing_columns) > 0:
        df_processed = df.drop(columns=existing_columns, axis=1)
        print(f"Silinen sütunlar: {existing_columns}")
        print(f"Kalan veri boyutu: {df_processed.shape}")
    else:
        df_processed = df.copy()
        print("Belirtilen sütunlar veri setinde bulunamadı.")
    
    return df_processed

# 4. Veri Ön İşleme
def preprocess_data(df, target_column):
    """Veri ön işleme."""
    # Hedef değişken var mı kontrol et
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun '{target_column}' veri setinde bulunamadı.")
    
    # Eksik değerleri kontrol et ve işle
    if df.isnull().sum().sum() > 0:
        print("Eksik değerler tespit edildi. İşleniyor...")
        # Sayısal sütunlar için ortalama ile doldur
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Kategorik sütunlar için mod ile doldur
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Hedef değişkeni ayır
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Kategorik değişkenleri kodla
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Eğitim veri boyutu: {X_train.shape}")
    print(f"Test veri boyutu: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, encoders

# 5. Sınıflandırma Modellerini Eğitme (Başarılı/Başarısız Tahmini)
def train_ml_models(X_train, X_test, y_train, y_test, problem_type='classification'):
    """Çeşitli ML modelleri eğit ve değerlendir."""
    if problem_type == 'classification':
        # Sınıflandırma modelleri
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
    else:
        # Regresyon modelleri
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEğitiliyor: {name}")
        model.fit(X_train, y_train)
        
        # Tahminler
        y_pred = model.predict(X_test)
        
        # Metrikler
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} - Doğruluk: {accuracy:.4f}")
            print("\nSınıflandırma Raporu:")
            cr = classification_report(y_test, y_pred)
            print(cr)
            
            # Karmaşıklık matrisi
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} - Karmaşıklık Matrisi')
            plt.xlabel('Tahmin Edilen')
            plt.ylabel('Gerçek')
            plt.show()
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
            
            # Gerçek vs Tahmin grafiği
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Gerçek Değerler')
            plt.ylabel('Tahmin Edilen Değerler')
            plt.title(f'{name} - Gerçek vs Tahmin (R² = {r2:.4f})')
            plt.tight_layout()
            plt.show()
            
            results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'predictions': y_pred
            }
    
    return results

# 6. ANN Modeli (Başarılı/Başarısız Tahmini)
def build_ann_classification_model(input_dim, output_dim=1, learning_rate=0.001):
    """ANN sınıflandırma modeli oluştur."""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()
    
    return model

def train_ann_model(X_train_scaled, X_test_scaled, y_train, y_test, 
                    problem_type='classification', epochs=100, batch_size=32):
    """ANN modelini eğit ve değerlendir."""
    # Hedef değişken türünü kontrol et
    if problem_type == 'classification':
        # Benzersiz sınıf sayısını bul
        n_classes = len(np.unique(y_train))
        
        # İkili sınıflandırma mı çok sınıflı mı?
        if n_classes <= 2:
            output_dim = 1
            y_train_ann = y_train
            y_test_ann = y_test
        else:
            # Çok sınıflı için one-hot encoding
            from tensorflow.keras.utils import to_categorical
            output_dim = n_classes
            y_train_ann = to_categorical(y_train)
            y_test_ann = to_categorical(y_test)
            
        # Modeli oluştur
        model = build_ann_classification_model(X_train_scaled.shape[1], output_dim)
    else:
        # Regresyon modeli
        model = Sequential([
            Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Regresyon için tek çıkış
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        model.summary()
        
        y_train_ann = y_train
        y_test_ann = y_test
    
    # Erken durdurma
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Eğitim
    history = model.fit(
        X_train_scaled, y_train_ann,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Eğitim geçmişi grafiği
    plt.figure(figsize=(12, 5))
    
    if problem_type == 'classification':
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Eğitim Kaybı')
        plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
        plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk')
        plt.legend()
    else:
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Eğitim Kaybı')
        plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Eğitim MAE')
        plt.plot(history.history['val_mae'], label='Doğrulama MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Tahminler
    y_pred_proba = model.predict(X_test_scaled)
    
    if problem_type == 'classification':
        if output_dim == 1:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_test_ann = np.argmax(y_test_ann, axis=1)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        print(f"ANN - Doğruluk: {accuracy:.4f}")
        print("\nSınıflandırma Raporu:")
        cr = classification_report(y_test, y_pred)
        print(cr)
        
        # Karmaşıklık matrisi
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('ANN - Karmaşıklık Matrisi')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.show()
        
        return model, accuracy, history
    else:
        y_pred = y_pred_proba.flatten()
        
        # Metrikler
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"ANN - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Gerçek vs Tahmin grafiği
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.title(f'ANN - Gerçek vs Tahmin (R² = {r2:.4f})')
        plt.tight_layout()
        plt.show()
        
        return model, r2, rmse, history

# 7. Özellik Önem Grafiği
def plot_feature_importance(model, feature_names):
    """Özellik önem grafiğini çiz."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Özellik Önemi')
        plt.title('Özellik Önem Sıralaması')
        plt.tight_layout()
        plt.show()
        
        # En önemli özellikleri yazdır
        print("En önemli 10 özellik:")
        for i in range(min(10, len(indices))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return importances

# 8. Hiperparametre Optimizasyonu
def optimize_hyperparameters(X_train, y_train, model_type='random_forest', problem_type='classification'):
    """Hiperparametre optimizasyonu."""
    if problem_type == 'classification':
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            scoring = 'accuracy'
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            scoring = 'accuracy'
        else:
            raise ValueError("Desteklenmeyen model tipi")
    else:
        # Regresyon için
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            scoring = 'r2'
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            scoring = 'r2'
        else:
            raise ValueError("Desteklenmeyen model tipi")
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"En iyi parametreler: {grid_search.best_params_}")
    if problem_type == 'classification':
        print(f"En iyi Doğruluk Skoru: {grid_search.best_score_:.4f}")
    else:
        print(f"En iyi R² Skoru: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 9. Sonuçları kaydet
def save_results(df_processed, file_path):
    """İşlenmiş veriyi kaydet."""
    df_processed.to_csv(file_path, index=False)
    print(f"İşlenmiş veri kaydedildi: {file_path}")

# 10. Model Karşılaştırma Grafiği
def plot_model_comparison(results, problem_type='classification'):
    """Model performanslarını karşılaştıran grafik çiz."""
    plt.figure(figsize=(12, 6))
    
    if problem_type == 'classification':
        # Doğruluk skorlarını çıkar
        models = list(results.keys())
        accuracy_scores = [results[model]['accuracy'] for model in models]
        
        # Grafik
        bars = plt.bar(models, accuracy_scores, color='skyblue')
        plt.title('Model Karşılaştırma - Doğruluk Skorları')
        plt.xlabel('Model')
        plt.ylabel('Doğruluk Skoru')
        plt.xticks(rotation=45)
        
        # Çubukların üzerine değerleri yaz
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
    else:
        # R² ve RMSE skorlarını çıkar
        models = list(results.keys())
        r2_scores = [results[model]['r2'] for model in models]
        rmse_scores = [results[model]['rmse'] for model in models]
        
        # R² grafiği
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(models, r2_scores, color='skyblue')
        plt.title('Model Karşılaştırma - R² Skorları')
        plt.xlabel('Model')
        plt.ylabel('R² Skoru')
        plt.xticks(rotation=45)
        
        # Çubukların üzerine değerleri yaz
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # RMSE grafiği
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(models, rmse_scores, color='salmon')
        plt.title('Model Karşılaştırma - RMSE Skorları')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # Çubukların üzerine değerleri yaz
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Ana fonksiyon
def main():
    # Dosya yolu
    file_path = "ogrenci_verileri.csv"  # CSV dosyanızın adını buraya girin
    
    # Silinecek sütunlar
    columns_to_drop = ["finalgrade", "exam_score", "gender", "age", "student_id"]  # Silmek istediğiniz sütunlar
    
    # Hedef değişken
    target_column = "success"  # Tahmin etmek istediğiniz sütun adı (başarılı/başarısız)
    
    # Problem tipi (classification veya regression)
    problem_type = "classification"
    
    # 1. Veriyi yükle
    df = load_data(file_path)
    
    # 2. Veri analizi
    analyze_data(df)
    
    # 3. Gereksiz sütunları sil
    df_processed = drop_columns(df, columns_to_drop)
    
    # 4. Veri ön işleme
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, encoders = preprocess_data(df_processed, target_column)
    
    # 5. ML modellerini eğit
    ml_results = train_ml_models(X_train, X_test, y_train, y_test, problem_type)
    
    # 6. Model karşılaştırma grafiği
    plot_model_comparison(ml_results, problem_type)
    
    # 7. En iyi modeli seç
    if problem_type == 'classification':
        best_model_name = max(ml_results, key=lambda x: ml_results[x]['accuracy'])
        best_metric = ml_results[best_model_name]['accuracy']
        print(f"\nEn iyi model: {best_model_name} (Doğruluk = {best_metric:.4f})")
    else:
        best_model_name = max(ml_results, key=lambda x: ml_results[x]['r2'])
        best_metric = ml_results[best_model_name]['r2']
        print(f"\nEn iyi model: {best_model_name} (R² = {best_metric:.4f})")
    
    best_model = ml_results[best_model_name]['model']
    
    # 8. Özellik önemini göster (RandomForest veya GradientBoosting için)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importances = plot_feature_importance(best_model, X_train.columns)
    
    # 9. Hiperparametre optimizasyonu
    if best_model_name == 'Random Forest':
        optimized_model = optimize_hyperparameters(X_train, y_train, 'random_forest', problem_type)
    elif best_model_name == 'Gradient Boosting':
        optimized_model = optimize_hyperparameters(X_train, y_train, 'gradient_boosting', problem_type)
    
    # 10. ANN modelini eğit
    if problem_type == 'classification':
        ann_model, ann_accuracy, history = train_ann_model(
            X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
        ann_metric = ann_accuracy
    else:
        ann_model, ann_r2, ann_rmse, history = train_ann_model(
            X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
        ann_metric = ann_r2
    
    # 11. İşlenmiş veriyi kaydet
    save_results(df_processed, "islenmis_ogrenci_verileri.csv")
    # 12. Sonuçları karşılaştır
print("\nModel Karşılaştırma:")
if problem_type == 'classification':
    for name, result in ml_results.items():
        print(f"{name}: Doğruluk = {result['accuracy']:.4f}")
    print(f"ANN: Doğruluk = {ann_results['accuracy']:.4f}")
else:
    for name, result in ml_results.items():
        print(f"{name}: Hata (MSE) = {result['mse']:.4f}")
    print(f"ANN: Hata (MSE) = {ann_results['mse']:.4f}")
