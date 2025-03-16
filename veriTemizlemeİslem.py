import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    # Veri setini yükleme
    df = pd.read_csv(file_path)
    
    # 1. Gereksiz sütunları kaldırma
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])
    if "Gender" in df.columns:
        df = df.drop(columns=["Gender"])
    
    # 2. Eksik verileri kontrol etme
    missing_values = df.isnull().sum()
    print("Eksik değerler:")
    print(missing_values)
    
    # 3a. İkili kategorik değişkenleri sayısal hale getirme
    binary_mappings = {
        "Gender": {"Male": 1, "Female": 0},
        "Participation_in_Discussions": {"Yes": 1, "No": 0},
        "Use_of_Educational_Tech": {"Yes": 1, "No": 0}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # 3b. Çoklu kategorik değişkenleri One-Hot Encoding ile dönüştürme
    categorical_columns = ["Preferred_Learning_Style", "Self_Reported_Stress_Level", "Final_Grade"]
    df = pd.get_dummies(df, columns=[col for col in categorical_columns if col in df.columns], drop_first=True)
    
    # 4. Sayısal verileri ölçeklendirme
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_scaled

# Kullanım
file_path = "student_performance_large_dataset.csv"  # Veri seti dosya yolu
df_processed = preprocess_data(file_path)
print("Ön işleme tamamlandı. İşlenmiş veri seti hazır.")
