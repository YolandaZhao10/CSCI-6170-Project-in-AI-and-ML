import argparse
import os
import time
import random
import numpy as np
import torch
import torch.optim as optim

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
from data import get_multidomain_dataloaders
from models import FeatureExtractor, LabelPredictor, DomainClassifier
from train import train_epoch
from evaluate import evaluate_accuracy, plot_confusion_matrix, plot_tsne, evaluate_mc_dropout

def main():
    parser = argparse.ArgumentParser(description='Multi-Domain Cross-Generalization Testbed')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['baseline', 'coral', 'dann', 'mc_dropout', 'irm', 'target_supervised', 'fine_tuning'],
                        help='Which method to run')
    parser.add_argument('--seed', type=int, default=42, help='Random deterministic seed')
    parser.add_argument('--source', type=str, default='MNIST', choices=['MNIST', 'USPS', 'SVHN'], help='Source domain dataset')
    parser.add_argument('--targets', type=str, nargs='+', default=['USPS', 'SVHN'], help='Target domain dataset(s)')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--penalty_weight', type=float, default=1.0, help='Weight for adaptation penalties')
    parser.add_argument('--dropout_p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--mc_passes', type=int, default=10, help='Number of MC Dropout passes')
    parser.add_argument('--irm_warmup_epochs', type=int, default=0, help='Epochs to wait before applying IRM penalty')
    parser.add_argument('--dann_gamma', type=float, default=10.0, help='DANN gradient reversal curve steepness')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Outputs dir
    os.makedirs('outputs', exist_ok=True)

    # 1. Data
    print(f"Loading dataloaders: Source={args.source}, Targets={args.targets}")
    src_tr, src_te, tgt_comb_tr, tgt_te_dict = get_multidomain_dataloaders(
        args.source, args.targets, batch_size=args.batch_size
    )

    # 2. Models
    use_dropout = (args.method == 'mc_dropout')
    feature_extractor = FeatureExtractor(use_dropout=use_dropout, dropout_p=args.dropout_p).to(device)
    label_predictor = LabelPredictor(use_dropout=use_dropout, dropout_p=args.dropout_p).to(device)
    
    domain_classifier = None
    params = list(feature_extractor.parameters()) + list(label_predictor.parameters())
    
    if args.method == 'dann':
        domain_classifier = DomainClassifier().to(device)
        params += list(domain_classifier.parameters())

    optimizer = optim.Adam(params, lr=args.lr)

    # 3. Training Loop
    print(f"\n--- Starting {args.method.upper()} Training ---")
    
    train_method = 'baseline' if args.method in ['target_supervised', 'fine_tuning'] else args.method
    active_src_tr = tgt_comb_tr if args.method == 'target_supervised' else src_tr
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss = train_epoch(train_method, feature_extractor, label_predictor, domain_classifier,
                           active_src_tr, tgt_comb_tr, optimizer, epoch, args.epochs,
                           device, args.penalty_weight,
                           irm_warmup_epochs=args.irm_warmup_epochs, dann_gamma=args.dann_gamma)
        epoch_time = time.time() - start_time
        
        src_acc = evaluate_accuracy(feature_extractor, label_predictor, src_te, device)
        tgt_accs = {name: evaluate_accuracy(feature_extractor, label_predictor, loader, device) for name, loader in tgt_te_dict.items()}
        
        tgt_str = " | ".join([f"Tgt [{k}]: {v:.4f}" for k, v in tgt_accs.items()])
        print(f"Epoch {epoch:02d} | Time: {epoch_time:.2f}s | Loss: {loss:.4f} | Src [{args.source}]: {src_acc:.4f} | {tgt_str}")

    if args.method == 'fine_tuning':
        print(f"\n--- Starting FT (Fine-Tuning on Target) ---")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1 # slower LR for fine tuning
            
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            loss = train_epoch('baseline', feature_extractor, label_predictor, None,
                               tgt_comb_tr, None, optimizer, epoch, args.epochs,
                               device, args.penalty_weight)
            epoch_time = time.time() - start_time
            
            src_acc = evaluate_accuracy(feature_extractor, label_predictor, src_te, device)
            tgt_accs = {name: evaluate_accuracy(feature_extractor, label_predictor, loader, device) for name, loader in tgt_te_dict.items()}
            
            tgt_str = " | ".join([f"Tgt [{k}]: {v:.4f}" for k, v in tgt_accs.items()])
            print(f"FT Epoch {epoch:02d} | Time: {epoch_time:.2f}s | Loss: {loss:.4f} | Src [{args.source}]: {src_acc:.4f} | {tgt_str}")

    # 4. Evaluation & Visualization per target
    print("\\n--- Generating Visualizations ---")
    classes = [str(i) for i in range(10)]
    
    for tgt_name, loader in tgt_te_dict.items():
        cm_path = f'outputs/cm_{args.method}_{args.source}_to_{tgt_name}.png'
        plot_confusion_matrix(feature_extractor, label_predictor, loader, classes, cm_path, device)
        print(f"Saved Confusion Matrix to {cm_path}")

        tsne_path = f'outputs/tsne_{args.method}_{args.source}_to_{tgt_name}.png'
        plot_tsne(feature_extractor, src_te, loader, tsne_path, device)
        print(f"Saved t-SNE plot to {tsne_path}")

        if args.method == 'mc_dropout':
            mcd_path = f'outputs/mc_dropout_var_{args.source}_to_{tgt_name}.png'
            evaluate_mc_dropout(feature_extractor, label_predictor, loader, device, passes=args.mc_passes, filepath=mcd_path)
            print(f"Saved MC Dropout Uncertainty Plot to {mcd_path}")

    print("Done!")

if __name__ == '__main__':
    main()
