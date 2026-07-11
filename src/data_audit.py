import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict
import warnings
from src.data_loader import get_toxcast_dataloaders, get_dataloaders
warnings.filterwarnings("ignore")

def audit_dataset(dataset_name="toxcast"):
    print(f"=== Running Data Audit for {dataset_name.upper()} ===")

    if dataset_name == 'toxcast':
        tr_l, va_l, te_l, pos_weights, num_tasks = get_toxcast_dataloaders(batch_size=1024)
    else:
        tr_l, va_l, te_l, pos_weights, num_tasks = get_dataloaders(batch_size=1024)

    def extract_stats(loader, name):
        all_y = []
        num_mols = 0
        for batch in loader:
            num_mols += batch.y.shape[0]
            all_y.append(batch.y.numpy())

        all_y = np.vstack(all_y)

        # Calculate positive and negative counts per task
        pos_counts = np.nansum(all_y == 1, axis=0)
        neg_counts = np.nansum(all_y == 0, axis=0)
        total_valid = pos_counts + neg_counts

        # Avoid division by zero
        safe_total = np.where(total_valid == 0, 1, total_valid)
        pos_ratio = pos_counts / safe_total

        print(f"\n[{name} Split]")
        print(f"Total Molecules: {num_mols}")
        print(f"Average Valid Labels per Endpoint: {np.mean(total_valid):.1f}")
        print(f"Average Positive Ratio across Endpoints: {np.mean(pos_ratio[total_valid > 0])*100:.2f}%")

        return {
            'num_mols': num_mols,
            'pos_counts': pos_counts,
            'neg_counts': neg_counts,
            'pos_ratio': pos_ratio,
            'total_valid': total_valid
        }

    tr_stats = extract_stats(tr_l, "Train")
    va_stats = extract_stats(va_l, "Validation")
    te_stats = extract_stats(te_l, "Test")

    # Check for severe imbalance issues (e.g. tasks with 0 positives in test)
    zero_pos_test = np.sum((te_stats['total_valid'] > 0) & (te_stats['pos_counts'] == 0))
    print(f"\n[Hygiene Check]")
    print(f"Endpoints with 0 positives in Test set: {zero_pos_test} / {num_tasks}")

    # Save detailed per-endpoint audit to CSV
    audit_df = pd.DataFrame({
        'Task_ID': np.arange(num_tasks),
        'Train_Pos': tr_stats['pos_counts'],
        'Train_Neg': tr_stats['neg_counts'],
        'Train_PosRatio': tr_stats['pos_ratio'],
        'Val_Pos': va_stats['pos_counts'],
        'Val_PosRatio': va_stats['pos_ratio'],
        'Test_Pos': te_stats['pos_counts'],
        'Test_PosRatio': te_stats['pos_ratio'],
    })

    os.makedirs('results', exist_ok=True)
    audit_df.to_csv(f'results/{dataset_name}_audit.csv', index=False)
    print(f"\nDetailed audit saved to results/{dataset_name}_audit.csv")

if __name__ == "__main__":
    # We will run this on a very tiny subset by setting an env var just to quickly test the script executes correctly,
    # but the full dataset takes 20-30m due to RDKit embed.
    import sys
    print("Script initialized.")
