import cv2
import os
import numpy as np
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import datetime
import sys

# --- Özellik çıkarım fonksiyonu ---
def extract_enhanced_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (800, 1000))  # normalize boyut

    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    total_pixels = binary.size
    black_pixels = total_pixels - cv2.countNonZero(binary)
    ink_ratio = black_pixels / total_pixels

    entropy_val = shannon_entropy(binary)

    edges = cv2.Canny(binary, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    num_labels, _ = cv2.connectedComponents(255 - binary)  # tersle -> siyahlar 255 olsun
    connected_components = num_labels

    return [ink_ratio, entropy_val, edge_density, connected_components]

# --- Eğitim veri dizinleri ---
def trainModel(empty_dir, filled_dir):
    # empty_dir = "train_pages/empty"
    # filled_dir = "train_pages/filled"
    X, y = [], []

    # Boş sayfaları işle
    for file in os.listdir(empty_dir):
        if file.lower().endswith(('.tif', '.tiff', '.jpg', '.png')):
            path = os.path.join(empty_dir, file)
            X.append(extract_enhanced_features(path))
            y.append(0)

    # Dolu sayfaları işle
    for file in os.listdir(filled_dir):
        if file.lower().endswith(('.tif', '.tiff', '.jpg', '.png')):
            path = os.path.join(filled_dir, file)
            X.append(extract_enhanced_features(path))
            y.append(1)

    # --- Model eğitimi ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # --- Değerlendirme ---
    y_pred = clf.predict(X_test)
    print("\n📊 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=["Boş", "Dolu"]))

    # --- Modeli kaydet ---
    joblib.dump(clf, "page_classifier_enhanced.pkl")
    print("✅ Model 'page_classifier_enhanced.pkl' olarak kaydedildi.\n")

# --- Örnek tahmin (test amaçlı) ---
def predict_page(image_path, model_path="page_classifier_enhanced.pkl"):
    model = joblib.load(model_path)
    feat = extract_enhanced_features(image_path)
    pred = model.predict([feat])[0]
    prob = model.predict_proba([feat])[0][1]
    prob = prob*100
    print(f"{pred},{prob:.2f}")  # stdout: örn: 1,93.27
    print(f"{os.path.basename(image_path)} ➜ Tahmin: {'Dolu' if pred == 1 else 'Boş'} (%{prob:.2f} güven)")

#trainModel("train_pages/empty","train_pages/filled")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("0,0.00")  # hata gibi davran
#         sys.exit(1)
#     predict_page(sys.argv[1])

# Test et (isteğe bağlı dosya yolları girin)
# predict_page("test_image.tif")


# print( datetime.datetime.now())
# # Kullanım örneği
predict_page("1.tif")
# # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
#
predict_page("2.tif")
# # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
#
predict_page("3.tif")
# # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
#
predict_page("4.tif")
# # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
#
predict_page("5.tif")
#
predict_page("6.tif")
#
# print( datetime.datetime.now())