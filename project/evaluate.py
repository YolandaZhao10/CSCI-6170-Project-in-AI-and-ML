import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import os

def evaluate_accuracy(feature_extractor, label_predictor, dataloader, device):
    feature_extractor.eval()
    label_predictor.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = feature_extractor(x)
            logits = label_predictor(features)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return correct / total

def plot_confusion_matrix(feature_extractor, label_predictor, dataloader, classes, filepath, device):
    feature_extractor.eval()
    label_predictor.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = feature_extractor(x)
            logits = label_predictor(features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

def plot_tsne(feature_extractor, source_loader, target_loader, filepath, device, num_samples=1000):
    feature_extractor.eval()
    features_list = []
    domain_labels = []
    
    with torch.no_grad():
        # Source features
        count = 0
        for x, _ in source_loader:
            x = x.to(device)
            feat = feature_extractor(x)
            features_list.append(feat.cpu().numpy())
            domain_labels.extend([0] * x.size(0))
            count += x.size(0)
            if count >= num_samples: break
            
        # Target features
        count = 0
        for x, _ in target_loader:
            x = x.to(device)
            feat = feature_extractor(x)
            features_list.append(feat.cpu().numpy())
            domain_labels.extend([1] * x.size(0))
            count += x.size(0)
            if count >= num_samples: break

    if len(features_list) == 0: return
    features_all = np.concatenate(features_list, axis=0)
    domain_all = np.array(domain_labels)
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_all)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=domain_all, cmap='coolwarm', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Source', 'Target'])
    plt.title('t-SNE of Extracted Features')
    plt.savefig(filepath)
    plt.close()

def evaluate_mc_dropout(feature_extractor, label_predictor, dataloader, device, passes=10, filepath='mc_dropout_uncertainty.png'):
    # Keep dropout active
    feature_extractor.train()
    label_predictor.train()
    
    all_preds_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # shape: (passes, batch_size, num_classes)
            batch_probs = []
            for _ in range(passes):
                features = feature_extractor(x)
                logits = label_predictor(features)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs.unsqueeze(0))
                
            batch_probs = torch.cat(batch_probs, dim=0) # passes x B x 10
            mean_probs = batch_probs.mean(dim=0)
            
            # Predictive Entropy: -sum(p * log(p))
            entropy = - torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            
            all_preds_probs.extend(mean_probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            plt.figure()
            plt.hist(entropy.cpu().numpy(), bins=20, alpha=0.7)
            plt.title('MC Dropout Predictive Entropy (1 Batch)')
            plt.xlabel('Predictive Entropy (Nats)')
            plt.ylabel('Frequency')
            plt.savefig(filepath)
            plt.close()
            break # Just do one batch for visualization
