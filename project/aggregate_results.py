import os
import glob
import re
import numpy as np
from collections import defaultdict

def parse_logs(log_dir="logs"):
    """
    Parses tuning and execution logs to extract final-epoch target accuracies.
    Expects log lines like:
    Epoch 10 | Time: 2.15s | Loss: 0.2341 | Src [MNIST]: 0.9811 | Tgt [USPS]: 0.7421 | Tgt [SVHN]: 0.2134
    """
    results = defaultdict(list)
    
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    for log_path in log_files:
        filename = os.path.basename(log_path)
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        # Extract metadata from filename (e.g. dann_MNIST_seed42_run.log or dann_srcMNIST_lr1e-3...)
        # A simple heuristic: method is the prefix, source is marked by _src or similar.
        # But we can also just parse the actual execution print statements to securely bind pairs.
        
        final_line = None
        for line in reversed(lines):
            if "Epoch" in line and "Src [" in line and "Tgt [" in line:
                final_line = line
                # prioritize FT Epochs if fine_tuning
                if "FT Epoch" in line:
                    break
        
        if final_line:
            method_match = re.search(r"Starting (\w+) Training", "".join(lines))
            method = method_match.group(1).lower() if method_match else "unknown"
            if method == 'baseline' and 'FT Epoch' in final_line:
                method = 'fine_tuning'
                
            src_match = re.search(r"Src \[([A-Za-z]+)\]: ([\d\.]+)", final_line)
            if src_match:
                source = src_match.group(1)
                
            # Find all targets
            tgt_matches = re.finditer(r"Tgt \[([A-Za-z]+)\]: ([\d\.]+)", final_line)
            for match in tgt_matches:
                target = match.group(1)
                acc = float(match.group(2))
                key = f"{method.upper()} | {source} -> {target}"
                results[key].append(acc)

    print("=" * 60)
    print(f"{'Method | Transfer Path':<35} | {'Mean ± Std Dev':<20}")
    print("=" * 60)
    
    for key in sorted(results.keys()):
        accs = np.array(results[key])
        mean = accs.mean() * 100
        std = accs.std() * 100
        print(f"{key:<35} | {mean:.2f}% ± {std:.2f}% (n={len(accs)})")

if __name__ == "__main__":
    parse_logs("logs")
