
import os
import sys

def main():
    print("Pre-commit checks...")

    # 1. Check if src directory exists
    if not os.path.exists("src"):
        print("Error: src directory missing")
        sys.exit(1)

    # 2. Check if main scripts exist
    if not os.path.exists("run_classical.py"):
        print("Error: run_classical.py missing")
        sys.exit(1)

    if not os.path.exists("run_quantum.py"):
        print("Error: run_quantum.py missing")
        sys.exit(1)

    # 3. Simple import check
    try:
        from src.data.loader import load_and_preprocess_data
        from src.features.graph import ToxDataset
        from src.models.gnn import MultiTaskGNN
        from src.models.quantum import HybridQGNN
        print("Imports successful.")
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    print("All checks passed.")

if __name__ == "__main__":
    main()
