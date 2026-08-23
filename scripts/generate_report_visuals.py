import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('report_figures', exist_ok=True)

# Load dynamically from results/benchmark_results.csv if it exists, else fallback to mock data
try:
    df = pd.read_csv('results/benchmark_results.csv')
    print("Loaded data from results/benchmark_results.csv")
except FileNotFoundError:
    print("results/benchmark_results.csv not found. Using mock data.")
    data = [
        [1, 'Quantum', 4, 51568, 0.5864, 0.3186, 0.2390],
        [1, 'Classical', 4, 196762, 0.6119, 0.3345, 0.2372],
        [2, 'Quantum', 4, 44732, 0.5773, 0.3124, 0.2497],
        [2, 'Classical', 4, 195380, 0.6169, 0.3335, 0.2476],
        [3, 'Quantum', 4, 44728, 0.5805, 0.3190, 0.2562],
        [3, 'Classical', 4, 142860, 0.6189, 0.3430, 0.2521],
        [1, 'Quantum', 6, 55696, 0.5759, 0.3129, 0.2374],
        [1, 'Classical', 6, 209038, 0.6004, 0.3224, 0.2391],
        [2, 'Quantum', 6, 46368, 0.5819, 0.3137, 0.2500],
        [2, 'Classical', 6, 199866, 0.6286, 0.3462, 0.2465],
        [3, 'Quantum', 6, 46362, 0.5906, 0.3173, 0.2384],
        [3, 'Classical', 6, 146114, 0.6227, 0.3419, 0.2466]
    ]
    df = pd.DataFrame(data, columns=['Level', 'Type', 'Qubits', 'Params', 'Test_ROC', 'Test_PR', 'Test_Brier'])

# Figure 1: Parameter Efficiency Scatter (using PR AUC now)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Params', y='Test_PR', hue='Type', style='Level', s=200)
plt.title('Parameter Efficiency: Test PR AUC vs Parameter Count', fontsize=14)
plt.xlabel('Number of Parameters')
plt.ylabel('Test PR AUC (Higher is Better)')
plt.grid(True, alpha=0.5)
plt.savefig('report_figures/parameter_efficiency.png')
plt.close()

# Figure 2: Qubit Scaling Trajectory (using PR AUC now)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Qubits', y='Test_PR', hue='Type', style='Level', markers=True, dashes=False, linewidth=2)
plt.title('Scaling Trajectory: Qubits vs Test PR AUC', fontsize=14)
plt.xlabel('Qubits (Quantum) / Parameter Scale (Classical)')
plt.ylabel('Test PR AUC (Higher is Better)')
plt.xticks([4, 6])
plt.grid(True, alpha=0.5)
plt.savefig('report_figures/scaling_trajectory.png')
plt.close()

# Figure 3: Performance Gap (Classical - Quantum PR AUC at max qubits)
max_qubits = df['Qubits'].max()
df_max = df[df['Qubits'] == max_qubits].copy()
classical_pr = df_max[df_max['Type'] == 'Classical'].set_index('Level')['Test_PR']
quantum_pr = df_max[df_max['Type'] == 'Quantum'].set_index('Level')['Test_PR']
gap = classical_pr - quantum_pr

plt.figure(figsize=(8, 6))
sns.barplot(x=gap.index, y=gap.values, palette='Blues')
plt.title(f'Absolute Performance Gap (Classical PR - Quantum PR)\nat {max_qubits} Qubits (Lower is Better for Quantum)', fontsize=14)
plt.xlabel('Inductive Bias Level')
plt.ylabel('PR AUC Gap')
for i, v in enumerate(gap.values):
    plt.text(i, v + 0.001, f"{v:.3f}", ha='center', va='bottom')
plt.savefig('report_figures/performance_gap.png')
plt.close()

# Figure 4: Calibration Comparison (Brier Score)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_max, x='Level', y='Test_Brier', hue='Type', palette='Set2')
plt.title(f'Model Calibration: Test Brier Score at {max_qubits} Qubits\n(Lower Brier Score is Better)', fontsize=14)
plt.xlabel('Inductive Bias Level')
plt.ylabel('Test Brier Score')
plt.grid(True, alpha=0.5, axis='y')
# Adjust legend position to not cover bars
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('report_figures/brier_score_comparison.png')
plt.close()

print("Figures generated in report_figures/")
