# ==========================================
# Bee vs Wasp Classification - Traditional Features + Robustness Testing
# ==========================================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent blocking plots
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Load Dataset
# -----------------------------

def load_images_from_folder(folder, label, img_size=(256,256)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = cv2.equalizeHist(img)
            data.append((img, label))
    return data

def load_dataset(base_dir, img_size=(256,256)):
    dataset = []
    classes = os.listdir(base_dir)
    for class_label in classes:
        folder_path = os.path.join(base_dir, class_label)
        if os.path.isdir(folder_path):
            dataset.extend(load_images_from_folder(folder_path, class_label, img_size))
    return dataset

# Path to dataset
base_dir = "bee_wasp/kaggle_bee_vs_wasp"
data = load_dataset(base_dir)

print("Total samples loaded:", len(data))
print("Classes:", set([label for _, label in data]))

# Split
X = [img for img, _ in data]
y = [label for _, label in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
class_names = le.classes_

# -----------------------------
# Step 2: Feature Extraction
# -----------------------------

def extract_hog_features(images):
    features = []
    for img in images:
        hog_feat = hog(img, orientations=9, pixels_per_cell=(16,16),
                       cells_per_block=(2,2), block_norm='L2-Hys')
        features.append(hog_feat)
    return np.array(features)

def extract_lbp_features(images, radius=3, n_points=24):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_points+3),
                               range=(0, n_points+2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)

def extract_edge_features(images, bins=16):
    features = []
    for img in images:
        edges = cv2.Canny(img, 100, 200)
        hist, _ = np.histogram(edges.ravel(), bins=bins, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)

# -----------------------------
# Step 3: Evaluation
# -----------------------------

def evaluate_model(X_train, X_test, y_train, y_test, method_name):
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "SVM (RBF)": SVC(kernel='rbf', probability=True)
    }

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n========== {method_name} ==========")
    results = {}
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            except:
                roc_auc = None
        else:
            roc_auc = None

        print(f"\n--- {clf_name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"Cohen’s Kappa: {kappa:.4f}, MCC: {mcc:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f"{method_name} - {clf_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        filename = f"{method_name}_{clf_name}_cm.png".replace(" ", "_")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        results[clf_name] = (clf, scaler, acc)

    return results

# -----------------------------
# Step 4: Robustness Testing
# -----------------------------

def add_noise(img, noise_type="gaussian"):
    if noise_type == "gaussian":
        row,col = img.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean,sigma,(row,col))
        noisy = img + gauss
        return np.clip(noisy,0,255).astype(np.uint8)
    elif noise_type == "blur":
        return cv2.GaussianBlur(img,(5,5),0)
    elif noise_type == "rotate":
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
        return cv2.warpAffine(img,M,(cols,rows))
    return img

def robustness_test(X_test, y_test, feature_func, trained_models, method_name):
    noise_types = ["gaussian", "blur", "rotate"]
    for noise in noise_types:
        X_test_noisy = [add_noise(img, noise) for img in X_test]
        X_test_feat = feature_func(X_test_noisy)

        for clf_name, (clf, scaler, _) in trained_models.items():
            X_test_scaled = scaler.transform(X_test_feat)
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            print(f"[Robustness] {method_name} - {clf_name} under {noise}: Accuracy = {acc:.4f}")

# -----------------------------
# Step 5: Run Experiments
# -----------------------------

# HOG
hog_train = extract_hog_features(X_train)
hog_test  = extract_hog_features(X_test)
hog_models = evaluate_model(hog_train, hog_test, y_train, y_test, "HOG Features")
robustness_test(X_test, y_test, extract_hog_features, hog_models, "HOG Features")

# LBP
lbp_train = extract_lbp_features(X_train)
lbp_test  = extract_lbp_features(X_test)
lbp_models = evaluate_model(lbp_train, lbp_test, y_train, y_test, "LBP Features")
robustness_test(X_test, y_test, extract_lbp_features, lbp_models, "LBP Features")

# Edge Detection
edge_train = extract_edge_features(X_train)
edge_test  = extract_edge_features(X_test)
edge_models = evaluate_model(edge_train, edge_test, y_train, y_test, "Edge Detection Features")
robustness_test(X_test, y_test, extract_edge_features, edge_models, "Edge Detection Features")
