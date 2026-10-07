import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt

def evaluate_models(base_dir="coding - models"):
    models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    results = []
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        folds = [d for d in os.listdir(model_dir) if d.startswith('fold')]
        
        all_y_true = []
        all_y_probs = []
        all_y_pred = []
        
        for fold in folds:
            fold_dir = os.path.join(model_dir, fold)
            y_true_path = os.path.join(fold_dir, "y_true.npy")
            y_probs_path = os.path.join(fold_dir, "y_pred_probs.npy")
            
            if os.path.exists(y_true_path) and os.path.exists(y_probs_path):
                y_true = np.load(y_true_path)
                y_probs = np.load(y_probs_path)
                y_pred = np.argmax(y_probs, axis=1)
                
                all_y_true.extend(y_true)
                all_y_probs.extend(y_probs)
                all_y_pred.extend(y_pred)
                
        if len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_probs = np.array(all_y_probs)
            all_y_pred = np.array(all_y_pred)
            
            acc = accuracy_score(all_y_true, all_y_pred)
            f1 = f1_score(all_y_true, all_y_pred, average='macro')
            prec = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
            rec = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(all_y_true, all_y_probs, multi_class='ovr', average='macro')
            except ValueError:
                auc = 0.0
                
            results.append({
                "Model": model,
                "Accuracy": acc,
                "Macro F1": f1,
                "Precision": prec,
                "Recall": rec,
                "AUC": auc
            })
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = evaluate_models()
    print("\n=======================================================")
    print("5-FOLD CROSS VALIDATION COMBINED RESULTS")
    print("=======================================================")
    print(df.to_string(index=False))
    df.to_csv("coding_models_comparison.csv", index=False)
    print("\nSaved comparison to coding_models_comparison.csv")
