import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')


def plot_confusion_matrix(cm, labels, normalize=False, title=None):
    """
    cm: ndarray (2x2)
    labels: sınıf adları
    normalize: satır bazlı normalize et
    title: grafik başlığı
    """
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel='Tahmin', ylabel='Gerçek', title=title
    )
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    return fig


class StudentPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Öğrenci Başarı Tahmini")
        self.root.geometry("1000x700")
        
        # GUI bileşenleri oluştur
        self.create_widgets()
        
        # Model ve veri değişkenleri
        self.df = None
        self.X = None
        self.y = None
        self.gbm_pipe = None
        self.lr_pipe = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_cols = []
        self.cat_cols = []
        self.num_ranges = {}
        self.cat_options = {}
        self.gbm_opt_threshold = 0.5
        self.lr_opt_threshold = 0.5
        self.feature_entries = {}
        
        # Veriyi yükle ve modeli eğit
        self.load_data_and_train_model()
    
    def create_widgets(self):
        # Ana notebook (sekmeli arayüz) oluştur
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Model değerlendirme sekmesi
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model Değerlendirme")
        
        # Tahmin sekmesi
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_tab, text="Yeni Öğrenci Tahmini")
        
        # --- Model değerlendirme sekmesi içeriği ---
        # Model bilgi paneli
        self.model_info_frame = ttk.LabelFrame(self.model_tab, text="Model Bilgisi")
        self.model_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_info_text = tk.Text(self.model_info_frame, height=10, width=80)
        self.model_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Confusion Matrix için frame ve canvas
        self.cm_frame = ttk.LabelFrame(self.model_tab, text="Karışıklık Matrisi")
        self.cm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.cm_canvas_frame = ttk.Frame(self.cm_frame)
        self.cm_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Tahmin sekmesi içeriği ---
        # Frame ve scrollbar oluştur
        self.predict_main_frame = ttk.Frame(self.predict_tab)
        self.predict_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sol taraf - öğrenci girişi
        self.input_frame = ttk.LabelFrame(self.predict_main_frame, text="Öğrenci Özellikleri")
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Input form için scrollbar ve canvas
        self.input_canvas = tk.Canvas(self.input_frame)
        self.input_scrollbar = ttk.Scrollbar(self.input_frame, orient="vertical", command=self.input_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.input_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.input_canvas.configure(scrollregion=self.input_canvas.bbox("all"))
        )
        
        self.input_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.input_canvas.configure(yscrollcommand=self.input_scrollbar.set)
        
        self.input_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tahmin butonu
        self.predict_btn = ttk.Button(self.input_frame, text="Tahmin Et", command=self.predict_student)
        self.predict_btn.pack(side=tk.BOTTOM, pady=10)
        
        # Sağ taraf - tahmin sonucu
        self.result_frame = ttk.LabelFrame(self.predict_main_frame, text="Tahmin Sonucu")
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.result_label = ttk.Label(self.result_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)
    
    def load_data_and_train_model(self):
        try:
            # --- Veri yükleme ve hazırlık ---
            self.df = pd.read_csv('student_performance_updated_v3.csv')
            self.df.drop(columns=['Student_ID', 'Final_Grade', 'Gender'], inplace=True)
            self.X = self.df.drop(columns=['Exam_Score (%)'])
            self.y = (self.df['Exam_Score (%)'] >= 50).astype(int)

            # Özellik türleri
            self.num_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.cat_cols = self.X.select_dtypes(include=['object']).columns.tolist()
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols)
            ], remainder='passthrough')

            # Eğitim/Test bölünmesi
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )

            # Sınıf ağırlıkları (dengesiz veri için)
            counts = self.y_train.value_counts(normalize=True)
            sample_weights = self.y_train.map({0: counts[1], 1: counts[0]}).values

            # --- GradientBoostingClassifier ---
            gbc = GradientBoostingClassifier(random_state=42)
            self.gbm_pipe = Pipeline([('pre', preprocessor), ('model', gbc)])
            self.gbm_pipe.fit(self.X_train, self.y_train, model__sample_weight=sample_weights)

            # --- LogisticRegression ---
            lr = LogisticRegression(
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                max_iter=500,
                random_state=42
            )
            self.lr_pipe = Pipeline([('pre', preprocessor), ('model', lr)])
            self.lr_pipe.fit(self.X_train, self.y_train)

            # Olasılık tahmini ve eşik optimizasyonu - GBM
            gbm_y_proba = self.gbm_pipe.predict_proba(self.X_test)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 17)
            gbm_f1_scores = [f1_score(self.y_test, (gbm_y_proba >= t).astype(int)) for t in thresholds]
            self.gbm_opt_threshold = thresholds[np.argmax(gbm_f1_scores)]
            
            # Olasılık tahmini ve eşik optimizasyonu - LogisticRegression
            lr_y_proba = self.lr_pipe.predict_proba(self.X_test)[:, 1]
            lr_f1_scores = [f1_score(self.y_test, (lr_y_proba >= t).astype(int)) for t in thresholds]
            self.lr_opt_threshold = thresholds[np.argmax(lr_f1_scores)]
            
            # Model bilgilerini göster
            self.display_model_info(gbm_y_proba, lr_y_proba)
            
            # Öğrenci özellik giriş formunu oluştur
            self.create_feature_inputs()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Veri yükleme veya model eğitimi sırasında hata: {str(e)}")
            
    def display_model_info(self, gbm_y_proba, lr_y_proba):
        # Model bilgilerini text alanına yaz
        self.model_info_text.delete(1.0, tk.END)
        
        # GBM Model Bilgileri
        self.model_info_text.insert(tk.END, "=== Gradyan Artırma Sınıflandırıcı ===\n")
        self.model_info_text.insert(tk.END, f"Optimal Eşik: {self.gbm_opt_threshold:.2f}, ")
        
        # F1 puanını ve doğruluk hesapla - GBM
        gbm_opt_preds = (gbm_y_proba >= self.gbm_opt_threshold).astype(int)
        gbm_default_preds = self.gbm_pipe.predict(self.X_test)
        gbm_max_f1 = f1_score(self.y_test, gbm_opt_preds)
        gbm_accuracy = np.mean(gbm_default_preds == self.y_test)
        self.model_info_text.insert(tk.END, f"F1={gbm_max_f1:.4f}, Doğruluk={gbm_accuracy:.4f}\n\n")
        
        # LogisticRegression Model Bilgileri
        self.model_info_text.insert(tk.END, "=== Lojistik Regresyon (Dengeli) ===\n")
        self.model_info_text.insert(tk.END, f"Optimal Eşik: {self.lr_opt_threshold:.2f}, ")
        
        # F1 puanını ve doğruluk hesapla - LogisticRegression
        lr_opt_preds = (lr_y_proba >= self.lr_opt_threshold).astype(int)
        lr_default_preds = self.lr_pipe.predict(self.X_test)
        lr_max_f1 = f1_score(self.y_test, lr_opt_preds)
        lr_accuracy = np.mean(lr_default_preds == self.y_test)
        self.model_info_text.insert(tk.END, f"F1={lr_max_f1:.4f}, Doğruluk={lr_accuracy:.4f}\n\n")
        
        # GBM için karışıklık matrisi ve raporları
        for label, preds in [('Varsayılan', self.gbm_pipe.predict(self.X_test)),
                            (f'Opt {self.gbm_opt_threshold:.2f}', gbm_opt_preds)]:
            cm = confusion_matrix(self.y_test, preds)
            report = classification_report(self.y_test, preds, target_names=['Başarısız','Başarılı'])
            
            self.model_info_text.insert(tk.END, f"Gradyan Artırma KM ({label}):\n{cm}\n")
            self.model_info_text.insert(tk.END, f"{report}\n")
            
            # Karışıklık matrisi görselleştirmelerini oluştur - GBM
            fig1 = plot_confusion_matrix(cm, ['Başarısız','Başarılı'], normalize=False, 
                                        title=f"GBM KM {label}")
            fig2 = plot_confusion_matrix(cm, ['Başarısız','Başarılı'], normalize=True, 
                                        title=f"GBM KM Norm {label}")
            
            # Grafikleri canvas'a ekle
            for fig in [fig1, fig2]:
                canvas = FigureCanvasTkAgg(fig, master=self.cm_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.LEFT, padx=5, pady=5)
        
        # LogisticRegression için karışıklık matrisi ve raporları
        for label, preds in [('Varsayılan', self.lr_pipe.predict(self.X_test)),
                            (f'Opt {self.lr_opt_threshold:.2f}', lr_opt_preds)]:
            cm = confusion_matrix(self.y_test, preds)
            report = classification_report(self.y_test, preds, target_names=['Başarısız','Başarılı'])
            
            self.model_info_text.insert(tk.END, f"Lojistik Regresyon KM ({label}):\n{cm}\n")
            self.model_info_text.insert(tk.END, f"{report}\n")
            
            # Karışıklık matrisi görselleştirmelerini oluştur - LR
            fig1 = plot_confusion_matrix(cm, ['Başarısız','Başarılı'], normalize=False, 
                                        title=f"LR KM {label}")
            fig2 = plot_confusion_matrix(cm, ['Başarısız','Başarılı'], normalize=True, 
                                        title=f"LR KM Norm {label}")
            
            # Grafikleri canvas'a ekle
            for fig in [fig1, fig2]:
                canvas = FigureCanvasTkAgg(fig, master=self.cm_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_feature_inputs(self):
        # Özellik aralıklarını ve kategorik seçenekleri hesapla
        self.num_ranges = {c: (self.df[c].min(), self.df[c].max()) for c in self.num_cols}
        self.cat_options = {c: sorted(self.df[c].dropna().unique()) for c in self.cat_cols}
        
        # Her özellik için giriş alanı oluştur
        row = 0
        for col in self.X.columns:
            label = ttk.Label(self.scrollable_frame, text=f"{col}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
            
            if col in self.num_cols:
                min_val, max_val = self.num_ranges[col]
                entry_var = tk.StringVar()
                entry = ttk.Entry(self.scrollable_frame, textvariable=entry_var, width=15)
                entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
                
                # Değer aralığını göster
                range_label = ttk.Label(self.scrollable_frame, 
                                      text=f"({min_val:.2f} - {max_val:.2f})",
                                      font=("Arial", 8))
                range_label.grid(row=row, column=2, sticky=tk.W, padx=5, pady=3)
                
                self.feature_entries[col] = entry_var
            
            elif col in self.cat_cols:
                combo_var = tk.StringVar()
                combo = ttk.Combobox(self.scrollable_frame, textvariable=combo_var, 
                                    values=self.cat_options[col], state="readonly", width=15)
                if len(self.cat_options[col]) > 0:
                    combo.current(0)  # İlk değeri seç
                combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
                self.feature_entries[col] = combo_var
            
            row += 1
    
    def prompt_student_features(self):
        """
        GUI'den öğrenci özelliklerini alır, aralıklarda olup olmadığını kontrol eder.
        """
        answers = {}
        for col in self.X.columns:
            if col in self.num_cols:
                min_val, max_val = self.num_ranges[col]
                try:
                    val = self.feature_entries[col].get().strip().replace(',', '.')
                    num = float(val)
                    if num < min_val or num > max_val:
                        messagebox.showwarning("Uyarı", 
                                             f"{col} değeri {min_val}–{max_val} aralığında olmalı.")
                        return None
                    answers[col] = num
                except ValueError:
                    messagebox.showwarning("Uyarı", f"{col} için geçerli bir sayı girin.")
                    return None
            elif col in self.cat_cols:
                val = self.feature_entries[col].get()
                if not val or val not in self.cat_options[col]:
                    messagebox.showwarning("Uyarı", f"{col} için geçerli bir seçenek seçin.")
                    return None
                answers[col] = val
        
        return pd.DataFrame([answers])
    
    def predict_student(self):
        try:
            # Öğrenci özelliklerini al
            X_new = self.prompt_student_features()
            if X_new is None:
                return
            
            # Sonuç alanını temizle
            self.result_label.config(text="")
            
            # Önceki grafikleri temizle
            for widget in self.result_frame.winfo_children():
                if widget != self.result_label:
                    widget.destroy()
            
            # Özellikleri doğru sırayla düzenle
            X_new = X_new[self.X.columns.tolist()]
            
            # GBM ile tahmin yap
            gbm_proba = self.gbm_pipe.predict_proba(X_new)[:, 1][0]
            gbm_opt_pred = (gbm_proba >= self.gbm_opt_threshold).astype(int)
            gbm_default_pred = self.gbm_pipe.predict(X_new)[0]
            
            # LogisticRegression ile tahmin yap
            lr_proba = self.lr_pipe.predict_proba(X_new)[:, 1][0]
            lr_opt_pred = (lr_proba >= self.lr_opt_threshold).astype(int)
            lr_default_pred = self.lr_pipe.predict(X_new)[0]
            
            # Sonucu göster
            result_text = "=== Gradyan Artırma Sınıflandırıcı ===\n"
            result_text += f"Tahmin (Varsayılan): {'BAŞARILI' if gbm_default_pred==1 else 'BAŞARISIZ'}\n"
            result_text += f"Tahmin (Opt {self.gbm_opt_threshold:.2f}): {'BAŞARILI' if gbm_opt_pred==1 else 'BAŞARISIZ'}\n"
            result_text += f"Başarı Olasılığı: {gbm_proba:.2f}\n\n"
            
            result_text += "=== Lojistik Regresyon ===\n"
            result_text += f"Tahmin (Varsayılan): {'BAŞARILI' if lr_default_pred==1 else 'BAŞARISIZ'}\n"
            result_text += f"Tahmin (Opt {self.lr_opt_threshold:.2f}): {'BAŞARILI' if lr_opt_pred==1 else 'BAŞARISIZ'}\n"
            result_text += f"Başarı Olasılığı: {lr_proba:.2f}"
            
            self.result_label.config(text=result_text)
            
            # Olasılık göstergelerini çiz
            self.display_probability_gauges(gbm_proba, lr_proba)
        except Exception as e:
            messagebox.showerror("Tahmin Hatası", f"Tahmin sırasında bir hata oluştu: {str(e)}")
    
    def display_probability_gauges(self, gbm_probability, lr_probability):
        try:
            # GBM Olasılık göstergesi
            fig_gbm, ax_gbm = plt.subplots(figsize=(5, 2))
            ax_gbm.barh(y=0, width=1, height=0.5, color='lightgray', alpha=0.3)
            ax_gbm.barh(y=0, width=gbm_probability, height=0.5, color=plt.cm.RdYlGn(gbm_probability))
            ax_gbm.axvline(x=self.gbm_opt_threshold, color='black', linestyle='--')
            ax_gbm.text(self.gbm_opt_threshold, 0.6, f'Eşik: {self.gbm_opt_threshold:.2f}', 
                     ha='center', rotation=90, fontsize=9)
            ax_gbm.text(0.05, 0, 'Başarısız', ha='left', va='center', fontsize=10)
            ax_gbm.text(0.95, 0, 'Başarılı', ha='right', va='center', fontsize=10)
            ax_gbm.text(gbm_probability, -0.3, f'{gbm_probability:.2f}', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7))
            ax_gbm.set_xlim(0, 1)
            ax_gbm.set_ylim(-0.5, 1)
            ax_gbm.set_title("GBM: Başarı Olasılığı")
            ax_gbm.set_yticks([])
            ax_gbm.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # LogisticRegression Olasılık göstergesi
            fig_lr, ax_lr = plt.subplots(figsize=(5, 2))
            ax_lr.barh(y=0, width=1, height=0.5, color='lightgray', alpha=0.3)
            ax_lr.barh(y=0, width=lr_probability, height=0.5, color=plt.cm.RdYlGn(lr_probability))
            ax_lr.axvline(x=self.lr_opt_threshold, color='black', linestyle='--')
            ax_lr.text(self.lr_opt_threshold, 0.6, f'Eşik: {self.lr_opt_threshold:.2f}', 
                    ha='center', rotation=90, fontsize=9)
            ax_lr.text(0.05, 0, 'Başarısız', ha='left', va='center', fontsize=10)
            ax_lr.text(0.95, 0, 'Başarılı', ha='right', va='center', fontsize=10)
            ax_lr.text(lr_probability, -0.3, f'{lr_probability:.2f}', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7))
            ax_lr.set_xlim(0, 1)
            ax_lr.set_ylim(-0.5, 1)
            ax_lr.set_title("Lojistik Regresyon: Başarı Olasılığı")
            ax_lr.set_yticks([])
            ax_lr.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Canvas'a ekle
            canvas_gbm = FigureCanvasTkAgg(fig_gbm, master=self.result_frame)
            canvas_gbm.draw()
            canvas_gbm.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            canvas_lr = FigureCanvasTkAgg(fig_lr, master=self.result_frame)
            canvas_lr.draw()
            canvas_lr.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("Grafik Hatası", f"Olasılık göstergesi oluşturulurken bir hata oluştu: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPredictorApp(root)
    root.mainloop()