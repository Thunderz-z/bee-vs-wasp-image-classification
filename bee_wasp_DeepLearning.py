import os
import time
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18, ResNet18_Weights,
    vgg16, VGG16_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score
)
from sklearn.model_selection import train_test_split
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.show()


def extract_features(model, dataloader, device, model_name='resnet18'):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            if model_name == 'resnet18':
                x = model.conv1(imgs)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                feats = torch.flatten(x, 1)


            elif model_name == 'vgg16':
                x = model.features(imgs)
                x = torch.flatten(x, 1)
                feats = model.classifier[0:4](x)  # Use first part of classifier as feature extractor


            elif model_name == 'mobilenetv2':
                x = model.features(imgs)
                x = torch.flatten(x, 1)
                feats = model.classifier[0:1](x)  # Single Linear layer usually, use as is


            else:
                raise ValueError(f"Unsupported model name: {model_name}")


            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())


    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


def evaluate_model(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=2000, multi_class='auto', solver='lbfgs')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)


    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")


    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovo')
        print(f"ROC-AUC (One-vs-One): {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC calculation failed: {e}")


    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    print(f"Cohen’s Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")


    error_mask = (y_pred != y_test)
    classes = np.unique(y_test)
    for cls in classes:
        total = np.sum(y_test == cls)
        errors = np.sum(error_mask & (y_test == cls))
        error_percent = 100.0 * errors / total if total > 0 else 0
        print(f"Class {cls}: {errors} errors out of {total} samples ({error_percent:.2f} %)")


def robustness_test(model, clf, dataloader, device, model_name):
    feature_list = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            if model_name == 'resnet18':
                x = model.conv1(imgs)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                feats = torch.flatten(x, 1)


            elif model_name == 'vgg16':
                x = model.features(imgs)
                x = torch.flatten(x, 1)
                feats = model.classifier[0:4](x)


            elif model_name == 'mobilenetv2':
                x = model.features(imgs)
                x = torch.flatten(x, 1)
                feats = model.classifier[0:1](x)


            feature_list.append(feats.cpu().numpy())


    features = np.concatenate(feature_list, axis=0)
    y_pred = clf.predict(features)
    return y_pred


def main():
    import multiprocessing
    multiprocessing.freeze_support()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    data_dir = 'bee_wasp/kaggle_bee_vs_wasp'
    img_size = 224


    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    dataset = datasets.ImageFolder(root=data_dir, transform=base_transform)
    print(f'Classes: {dataset.classes}')


    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


    # Gaussian blur transform for robustness test
    blur_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        GaussianBlur(kernel_size=5, sigma=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    dataset_blur = datasets.ImageFolder(root=data_dir, transform=blur_transform)
    dataloader_blur = DataLoader(dataset_blur, batch_size=32, shuffle=False, num_workers=4)


    # Models dictionary
    models_info = {
        'ResNet18': (resnet18(weights=ResNet18_Weights.DEFAULT).to(device), 'resnet18'),
        'VGG16': (vgg16(weights=VGG16_Weights.DEFAULT).to(device), 'vgg16'),
        'MobileNetV2': (mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device), 'mobilenetv2')
    }


    for model_name, (model, short_name) in models_info.items():
        print(f"\n\n########## {model_name} ##########")


        start_time = time.time()
        features, labels = extract_features(model, dataloader, device, short_name)
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, stratify=labels, test_size=0.2, random_state=42
        )


        clf = LogisticRegression(max_iter=2000, multi_class='auto', solver='lbfgs')
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"{model_name} - Feature Extraction + Classifier Training Time: {train_time:.2f}s")


        # Evaluation
        print(f"\n{model_name} Evaluation:")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovo')
            print(f"ROC-AUC (One-vs-One): {roc_auc:.4f}")
        except Exception as e:
            print(f"ROC-AUC calculation failed: {e}")

        # Plot confusion matrix heatmap
        plot_confusion_matrix(y_test, y_pred, dataset.classes, model_name)

        print(f"Cohen’s Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")


        error_mask = (y_pred != y_test)
        for class_idx, class_name in enumerate(dataset.classes):
            total = np.sum(y_test == class_idx)
            errors = np.sum(error_mask & (y_test == class_idx))
            error_percent = 100.0 * errors / total if total > 0 else 0
            print(f"Class '{class_name}': {errors} errors out of {total} samples ({error_percent:.2f} %)")


        # Robustness test on Gaussian blurred images
        y_pred_blur = robustness_test(model, clf, dataloader_blur, device, short_name)
        robust_acc = accuracy_score(labels, y_pred_blur)  # evaluating on all dataset here
        print(f"{model_name} Robustness Test Accuracy (Gaussian Blur): {robust_acc:.4f}")


if __name__ == '__main__':
    main()
