import torch
import torch.nn as nn
from tqdm import tqdm
from losses import coral_loss, penalty_irm

def train_epoch(method, feature_extractor, label_predictor, domain_classifier,
                source_loader, target_loader, optimizer, epoch, total_epochs,
                device, penalty_weight=1.0, irm_warmup_epochs=0, dann_gamma=10.0):
    
    feature_extractor.train()
    label_predictor.train()
    if domain_classifier:
        domain_classifier.train()

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    target_iter = iter(target_loader) if target_loader is not None else None
    total_loss = 0.0

    pbar = tqdm(source_loader, desc=f"Epoch {epoch}/{total_epochs}")
    for i, (source_x, source_y) in enumerate(pbar):
        source_x, source_y = source_x.to(device), source_y.to(device)
        
        target_x = None
        if target_iter is not None:
            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)
            target_x = target_x.to(device)

        if method in ['baseline', 'mc_dropout']:
            features = feature_extractor(source_x)
            logits = label_predictor(features)
            loss = criterion_class(logits, source_y)

        elif method == 'coral':
            source_features = feature_extractor(source_x)
            target_features = feature_extractor(target_x)

            logits = label_predictor(source_features)
            class_loss = criterion_class(logits, source_y)
            c_loss = coral_loss(source_features, target_features)

            loss = class_loss + penalty_weight * c_loss

        elif method == 'irm':
            half = source_x.size(0) // 2
            x1, y1 = source_x[:half], source_y[:half]
            x2, y2 = source_x[half:], source_y[half:]

            feat1 = feature_extractor(x1)
            feat2 = feature_extractor(x2)
            logits1 = label_predictor(feat1)
            logits2 = label_predictor(feat2)

            loss1 = criterion_class(logits1, y1)
            loss2 = criterion_class(logits2, y2)
            
            penalty1 = penalty_irm(logits1, y1)
            penalty2 = penalty_irm(logits2, y2)
            
            current_penalty = penalty_weight if epoch > irm_warmup_epochs else 0.0

            loss = loss1 + loss2 + current_penalty * (penalty1 + penalty2)

        elif method == 'dann':
            combined_x = torch.cat([source_x, target_x], dim=0)
            combined_features = feature_extractor(combined_x)
            
            source_features = combined_features[:source_x.size(0)]
            logits = label_predictor(source_features)
            class_loss = criterion_class(logits, source_y)
            
            p_val = float(i + epoch * len(source_loader)) / (total_epochs * len(source_loader))
            alpha = (2. / (1. + torch.exp(torch.tensor(-dann_gamma * p_val)))) - 1.
            
            domain_logits = domain_classifier(combined_features, alpha).squeeze()
            
            domain_labels = torch.cat([
                torch.zeros(source_x.size(0), device=device),
                torch.ones(target_x.size(0), device=device)
            ])
            domain_loss = criterion_domain(domain_logits, domain_labels)

            loss = class_loss + penalty_weight * domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(source_loader)
