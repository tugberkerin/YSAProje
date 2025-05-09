#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Farklı modellerin değerlendirmesi ve karşılaştırılması için fonksiyonlar içeren modül
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def model_performanslari_karsilastir(model_metrics_dict):
    """
    Farklı modellerin performanslarını karşılaştırır ve görselleştirir
    
    Parametreler:
    model_metrics_dict (dict): Model adları ve performans metriklerini içeren sözlük
                             Örnek: {'MLR': {'MSE': 0.1, 'RMSE': 0.3, 'MAE': 0.2, 'R2': 0.8},
                                     'ANN': {'MSE': 0.05, 'RMSE': 0.22, 'MAE': 0.18, 'R2': 0.9}}
    """
    print("Model performansları karşılaştırılıyor...")
    
    # Verileri DataFrame'e dönüştürme
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    models = list(model_metrics_dict.keys())
    
    comparison_data = []
    
    for model in models:
        model_metrics = model_metrics_dict[model]
        for metric in metrics:
            comparison_data.append({
                'Model': model,
                'Metrik': metric,
                'Değer': model_metrics[metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Metriklere göre gruplandırma
    grouped_df = comparison_df.pivot(index='Metrik', columns='Model', values='Değer')
    
    # Sonuçları yazdırma
    print("Model Karşılaştırma Sonuçları:")
    print(grouped_df)
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        metric_data = comparison_df[comparison_df['Metrik'] == metric]
        
        ax = axes[i]
        ax.bar(metric_data['Model'], metric_data['Değer'])
        ax.set_title(f'{metric} Karşılaştırması')
        ax.set_ylabel(metric)
        
        # R2 için daha yüksek değer daha iyi, diğerleri için daha düşük değer daha iyi
        if metric == 'R2':
            best_model = metric_data.loc[metric_data['Değer'].idxmax()]['Model']
            best_value = metric_data['Değer'].max()
        else:
            best_model = metric_data.loc[metric_data['Değer'].idxmin()]['Model']
            best_value = metric_data['Değer'].min()
            
        ax.text(best_model, best_value, f'{best_value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    print("Model karşılaştırma sonuçları görselleştirildi: results/model_comparison.png")
    
    # En iyi modeli belirleme
    best_models = {}
    for metric in metrics:
        metric_data = comparison_df[comparison_df['Metrik'] == metric]
        
        if metric == 'R2':  # R2 için daha yüksek değer daha iyi
            best_model = metric_data.loc[metric_data['Değer'].idxmax()]['Model']
            best_value = metric_data['Değer'].max()
        else:  # MSE, RMSE, MAE için daha düşük değer daha iyi
            best_model = metric_data.loc[metric_data['Değer'].idxmin()]['Model']
            best_value = metric_data['Değer'].min()
            
        best_models[metric] = (best_model, best_value)
    
    print("\nEn İyi Model Değerlendirmesi:")
    for metric, (model, value) in best_models.items():
        print(f"- {metric} metriğine göre en iyi model: {model} (Değer: {value:.4f})")
    
    return grouped_df, best_models

def feature_importance_karsilastir(mlr_importance, ann_importance):
    """
    MLR ve ANN modellerinin özellik önem analizlerini karşılaştırır
    
    Parametreler:
    mlr_importance (DataFrame): MLR modeli özellik önem analizi sonuçları
    ann_importance (DataFrame): ANN modeli özellik önem analizi sonuçları
    
    Döndürülen:
    DataFrame: Karşılaştırmalı özellik önem skorları
    """
    print("Özellik önem analizleri karşılaştırılıyor...")
    
    # MLR için normalize edilmiş önem skorları
    mlr_normalized = mlr_importance.copy()
    mlr_normalized['Normalize_Skor'] = mlr_normalized['Önem_Skoru'] / mlr_normalized['Önem_Skoru'].max()
    
    # ANN için normalize edilmiş önem skorları
    ann_normalized = ann_importance.copy()
    ann_normalized['Normalize_Skor'] = ann_normalized['Önem_Skoru'] / ann_normalized['Önem_Skoru'].max()
    
    # İki modelin özellik önem skorlarını birleştirme
    mlr_scores = mlr_normalized.set_index('Özellik')['Normalize_Skor']
    ann_scores = ann_normalized.set_index('Özellik')['Normalize_Skor']
    
    combined_scores = pd.DataFrame({
        'MLR_Önem': mlr_scores,
        'ANN_Önem': ann_scores
    }).reset_index()
    
    # Ortalama önem skoruna göre sıralama
    combined_scores['Ortalama_Önem'] = (combined_scores['MLR_Önem'] + combined_scores['ANN_Önem']) / 2
    combined_scores = combined_scores.sort_values('Ortalama_Önem', ascending=False)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(combined_scores))
    width = 0.35
    
    plt.bar(x - width/2, combined_scores['MLR_Önem'], width, label='MLR')
    plt.bar(x + width/2, combined_scores['ANN_Önem'], width, label='ANN')
    
    plt.xlabel('Özellik')
    plt.ylabel('Normalize Edilmiş Önem Skoru')
    plt.title('MLR ve ANN Modellerinin Özellik Önem Karşılaştırması')
    plt.xticks(x, combined_scores['Özellik'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    print("Karşılaştırmalı özellik önem analizi görselleştirildi: results/feature_importance.png")
    
    return combined_scores

def en_etkili_ozellikleri_belirle(combined_importance, top_n=5):
    """
    İki modelin sonuçlarına göre en etkili özellikleri belirler
    
    Parametreler:
    combined_importance (DataFrame): Karşılaştırmalı özellik önem skorları
    top_n (int): Kaç özelliğin seçileceği
    
    Döndürülen:
    list: En etkili özelliklerin listesi
    """
    print(f"En etkili {top_n} özellik belirleniyor...")
    
    top_features = combined_importance.head(top_n)['Özellik'].tolist()
    
    print(f"En etkili {top_n} özellik:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")
    
    return top_features

if __name__ == "__main__":
    print("Bu bir modül olarak tasarlanmıştır. Lütfen main.py üzerinden çalıştırın.")