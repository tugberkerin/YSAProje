import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def veri_setini_yukle(veri_yolu):
    """
    Veri setini yükler ve temel inceleme yapar
    
    Parametreler:
    veri_yolu (str): Veri seti dosyasının yolu
    
    Döndürülen:
    DataFrame: Yüklenen veri seti
    """
    print(f"{veri_yolu} yolundaki veri seti yükleniyor...")
    
    try:
        # Veri setini okuma
        df = pd.read_csv(veri_yolu)
        
        # Veri seti hakkında temel bilgiler
        print(f"Veri seti başarıyla yüklendi. Boyut: {df.shape}")
        print("\nİlk 5 satır:")
        print(df.head())
        
        print("\nVeri seti bilgileri:")
        print(df.info())
        
        print("\nİstatistiksel özet:")
        print(df.describe())
        
        # Eksik değerlerin kontrolü
        eksik_degerler = df.isnull().sum()
        if eksik_degerler.sum() > 0:
            print("\nEksik değerler tespit edildi:")
            print(eksik_degerler[eksik_degerler > 0])
        else:
            print("\nEksik değer bulunmamaktadır.")
        
        # Kategorik sütunları kontrol etme
        kategorik_sutunlar = df.select_dtypes(include=['object']).columns
        if len(kategorik_sutunlar) > 0:
            print("\nKategorik sütunlar:")
            for sutun in kategorik_sutunlar:
                print(f"{sutun}: {df[sutun].unique()}")
        
        # Hedef değişkenin dağılımını gösterme (Sınav Puanı)
        # Not: Hedef değişken adı veri setinize göre değişebilir
        hedef_degisken = None
        muhtemel_hedef_sutunlar = ["Exam_Score", "Final_Grade", "Sınav_Puanı"]
        
        for sutun in muhtemel_hedef_sutunlar:
            if sutun in df.columns:
                hedef_degisken = sutun
                break
        
        if hedef_degisken:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[hedef_degisken], kde=True)
            plt.title(f"{hedef_degisken} Dağılımı")
            plt.xlabel(hedef_degisken)
            plt.ylabel("Frekans")
            plt.savefig("results/hedef_degisken_dagilimi.png")
            print(f"\nHedef değişken ({hedef_degisken}) dağılımı görselleştirildi: results/hedef_degisken_dagilimi.png")
        
        return df
        
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

if __name__ == "__main__":
    # Bu modül doğrudan çalıştırıldığında test amaçlı olarak
    veri_setini_yukle("../data/student_performance_data.csv")