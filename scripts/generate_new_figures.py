import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('docs/figures', exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Tox21 vs BBBP Scaling Law
k_vals = [4, 6, 8]
tox21_dauc = [0.0078, 0.0108, 0.0134]
bbbp_dauc = [0.0070, 0.0130, 0.0180]

plt.figure(figsize=(6, 4))
plt.plot(k_vals, tox21_dauc, marker='o', linewidth=2, label='Tox21 (Nuclear Receptors)')
plt.plot(k_vals, bbbp_dauc, marker='s', linewidth=2, label='BBBP (Membrane Permeability)')
plt.xlabel("Number of Qubits / Clusters (K)")
plt.ylabel(r"Inductive Bias Margin ($\Delta$AUC)")
plt.title("Property P7: Substrate-Independent Scaling")
plt.xticks(k_vals)
plt.legend()
plt.tight_layout()
plt.savefig('docs/figures/fig_scaling_slopes.png', dpi=300)
plt.close()

# 2. Phase K Component Ablation
components = ["Baseline Level-8", "+ Ext. Observables", "+ Aromatic Gates", "Full QMP Framework"]
dauc_vals = [0.0100, 0.0120, 0.0148, 0.0218]

plt.figure(figsize=(7, 4))
bars = plt.barh(components, dauc_vals, color=sns.color_palette("Blues", len(components)))
plt.xlabel(r"Median $\Delta$AUC (Structured vs EGNN-Scrambled)")
plt.title("Phase K: Architecture Evolution Contributions")
for bar in bars:
    plt.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
             f'+{bar.get_width():.4f}', va='center', ha='left')
plt.xlim(0, 0.025)
plt.tight_layout()
plt.savefig('docs/figures/fig_phase_k_ablation.png', dpi=300)
plt.close()

# 3. Phase L Noise Resilience
noise_profiles = ["Ideal Statevector", "IBM Eagle\n(1% gate, 2% meas)", "Heavy NISQ\n(5% gate, 5% meas)"]
k4_retention = [100.0, 88.0, 29.0]
k6_retention = [100.0, 89.4, 29.8]

x = np.arange(len(noise_profiles))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
rects1 = ax.bar(x - width/2, k4_retention, width, label='K=4 Readout', color='lightcoral')
rects2 = ax.bar(x + width/2, k6_retention, width, label='K=6 Readout', color='firebrick')

ax.set_ylabel('Signal Margin Retention (%)')
ax.set_title('Phase L: QMP Resilience Under OQS Dynamics')
ax.set_xticks(x)
ax.set_xticklabels(noise_profiles)
ax.legend()
plt.tight_layout()
plt.savefig('docs/figures/fig_phase_l_noise.png', dpi=300)
plt.close()

# 4. MaxCut Optimization Trajectory
epochs = np.arange(20, 101, 20)
approx_ratio = [0.694, 0.906, 0.922, 0.923, 0.923]

plt.figure(figsize=(6, 4))
plt.plot(epochs, approx_ratio, marker='d', color='forestgreen', linewidth=2)
plt.axhline(y=1.0, color='r', linestyle='--', label='Exact Cut (Optimal)')
plt.xlabel("Training Epochs")
plt.ylabel("Approximation Ratio")
plt.title("Route C: QMP (L=2) on MaxCut (Erdos-Renyi, K=8)")
plt.legend()
plt.tight_layout()
plt.savefig('docs/figures/fig_maxcut_traj.png', dpi=300)
plt.close()

print("Figures successfully generated in docs/figures/")
