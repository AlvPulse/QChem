"""
E5 K-scaling convenience runner: extends the linear Delta-AUC law to K=10,12.
Run this script as: python run_e5_kscale.py [--full] [--preliminary]
Default: preliminary mode (--max_mols 500, --seeds 0 --folds 2)
--full: all mols, 3 seeds, 3 folds (compute-bound, hours)
"""
import subprocess, sys, argparse, pathlib

ap = argparse.ArgumentParser()
ap.add_argument('--full', action='store_true')
ap.add_argument('--preliminary', action='store_true')
args = ap.parse_args()

# Preliminary: fast check to see if K=10 bias persists
prelim_cmd = [
    sys.executable, 'run_levelG_probe.py',
    '--qubits', '10',
    '--configs', 'levelG', 'gate',
    '--seeds', '0',
    '--folds', '2',
    '--epochs', '20',
    '--max_mols', '500',
    '--datasets', 'Tox21', 'ToxCast',
    '--out', 'results/e5_kscale_prelim.json',
]

full_cmd = [
    sys.executable, 'run_levelG_probe.py',
    '--qubits', '8', '10', '12',
    '--configs', 'levelG', 'gate',
    '--seeds', '0', '1', '2',
    '--folds', '3',
    '--epochs', '30',
    '--datasets', 'Tox21', 'ToxCast',
    '--out', 'results/e5_kscale_full.json',
]

cmd = full_cmd if args.full else prelim_cmd
print('Running E5 K-scaling:', ' '.join(cmd))
subprocess.run(cmd, check=True)
