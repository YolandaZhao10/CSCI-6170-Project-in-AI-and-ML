import glob
import re
import os

def generate_summary():
    logs = glob.glob("logs/tuning/*.log")
    if not logs:
        print("No logs found in logs/tuning/")
        return
        
    results = []
    
    for log in logs:
        filename = os.path.basename(log).replace(".log", "")
        with open(log, "r") as f:
            content = f.read()
            
        # Extract target accuracies from the final epoch line
        lines = content.strip().split('\n')
        final_epoch = None
        for line in reversed(lines):
            if "Epoch " in line and "Tgt [" in line:
                final_epoch = line
                break
                
        if not final_epoch:
            continue
            
        # Extract source and target accuracies
        src_match = re.search(r"Src \[([A-Za-z]+)\]: ([\d\.]+)", final_epoch)
        src_name = src_match.group(1) if src_match else "Unknown"
        src_acc = src_match.group(2) if src_match else "N/A"
        
        tgts = re.findall(r"Tgt \[([A-Za-z]+)\]: ([\d\.]+)", final_epoch)
        
        for tgt_name, tgt_acc in tgts:
            results.append((filename, src_name, tgt_name, float(tgt_acc), float(src_acc)))
            
    # Sort results to find best configs easily
    results.sort(key=lambda x: (x[1], x[2], -x[3]))
    
    with open("tuning_summary.md", "w") as f:
        f.write("# Hyperparameter Tuning Summary\n\n")
        f.write("This document summarizes the best hyperparameter configurations discovered during the `tune.sh` sweep.\n\n")
        
        current_pair = None
        for r in results:
            run_name, src, tgt, tgt_acc, src_acc = r
            pair = f"{src} -> {tgt}"
            
            if pair != current_pair:
                f.write(f"\n## Transfer: {pair}\n")
                f.write("| Configuration Log Name | Target Acc | Source Acc |\n")
                f.write("| :--- | :---: | :---: |\n")
                current_pair = pair
                
            f.write(f"| `{run_name}` | **{tgt_acc*100:.2f}%** | {src_acc*100:.2f}% |\n")

if __name__ == "__main__":
    generate_summary()
