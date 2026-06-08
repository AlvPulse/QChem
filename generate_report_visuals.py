import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('report_figures', exist_ok=True)

# Generate Mock DataFrame from user provided data to guarantee we can run without requiring full retrain
data = [
    [1, 'Quantum', 4, 51568, 0.5864, 0.3186],
    [1, 'Classical', 4, 196762, 0.6119, 0.3345],
    [2, 'Quantum', 4, 44732, 0.5773, 0.3124],
    [2, 'Classical', 4, 195380, 0.6169, 0.3335],
    [3, 'Quantum', 4, 44728, 0.5805, 0.3190],
    [3, 'Classical', 4, 142860, 0.6189, 0.3430],
    [1, 'Quantum', 6, 55696, 0.5759, 0.3129],
    [1, 'Classical', 6, 209038, 0.6004, 0.3224],
    [2, 'Quantum', 6, 46368, 0.5819, 0.3137],
    [2, 'Classical', 6, 199866, 0.6286, 0.3462],
    [3, 'Quantum', 6, 46362, 0.5906, 0.3173],
    [3, 'Classical', 6, 146114, 0.6227, 0.3419]
]
df = pd.DataFrame(data, columns=['Level', 'Type', 'Qubits', 'Params', 'Test_ROC', 'Test_PR'])

# Figure 1: Parameter Efficiency Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Params', y='Test_ROC', hue='Type', style='Level', s=200)
plt.title('Parameter Efficiency: Test ROC vs Parameter Count', fontsize=14)
plt.xlabel('Number of Parameters')
plt.ylabel('Test ROC AUC')
plt.grid(True, alpha=0.5)
plt.savefig('report_figures/parameter_efficiency.png')
plt.close()

# Figure 2: Qubit Scaling Trajectory
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Qubits', y='Test_ROC', hue='Type', style='Level', markers=True, dashes=False, linewidth=2)
plt.title('Scaling Trajectory: Qubits vs Test ROC', fontsize=14)
plt.xlabel('Qubits (Quantum) / Parameter Scale (Classical)')
plt.ylabel('Test ROC AUC')
plt.xticks([4, 6])
plt.grid(True, alpha=0.5)
plt.savefig('report_figures/scaling_trajectory.png')
plt.close()

# Figure 3: Performance Gap (Classical - Quantum ROC at max qubits)
df_max = df[df['Qubits'] == 6].copy()
classical_roc = df_max[df_max['Type'] == 'Classical'].set_index('Level')['Test_ROC']
quantum_roc = df_max[df_max['Type'] == 'Quantum'].set_index('Level')['Test_ROC']
gap = classical_roc - quantum_roc

plt.figure(figsize=(8, 6))
sns.barplot(x=gap.index, y=gap.values, palette='Reds')
plt.title('Absolute Performance Gap (Classical ROC - Quantum ROC)\nat 6 Qubits (Lower is Better for Quantum)', fontsize=14)
plt.xlabel('Inductive Bias Level')
plt.ylabel('ROC Gap')
for i, v in enumerate(gap.values):
    plt.text(i, v + 0.001, f"{v:.3f}", ha='center', va='bottom')
plt.savefig('report_figures/performance_gap.png')
plt.close()

print("Figures generated in report_figures/")
