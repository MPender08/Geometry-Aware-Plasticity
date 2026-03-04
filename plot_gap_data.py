import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import EngFormatter

DATA_FILE = 'gap_learning_data.txt'

if not os.path.exists(DATA_FILE):
    print(f"Error: Data file '{DATA_FILE}' not found. Run NGSpice first.")
    exit()

# Load data (NGSpice wrdata outputs pairs of Time/Value columns for each variable)
data = np.loadtxt(DATA_FILE) 

time_sec = data[:, 0]
v_stdp = data[:, 1]     # The local STDP gradient
v_gamma = data[:, 3]    # The macroscopic curvature
v_phi = data[:, 5]      # The geometric multiplier
v_mem = data[:, 7]      # The memristor conductance state

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8), dpi=300)
plt.style.use('seaborn-v0_8-paper')

# Panel 1: Local vs Global State
ax1.plot(time_sec * 1e3, v_stdp, color='navy', label='STDP Gradient ($V_{stdp}$)', alpha=0.7)
ax1.plot(time_sec * 1e3, v_gamma, color='crimson', label='Macroscopic Gamma ($V_{\gamma}$)')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('A: Local Learning Gradient vs Global Network Curvature')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: The Geometric Multiplier
ax2.plot(time_sec * 1e3, v_phi, color='purple', linewidth=2)
ax2.set_ylabel('Multiplier Factor')
ax2.set_title('B: Geometric Multiplier ($\Phi$) Activation')
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel 3: Memristor State (The Structural Burn)
ax3.plot(time_sec * 1e3, v_mem, color='darkgreen', linewidth=2)
ax3.set_ylabel('Weight State (V)')
ax3.set_xlabel('Time (ms)')
ax3.set_title('C: Memristor Conductance (Structural Burn & Saturation)')
ax3.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('gap_learning_results.png')
print("Figure generated: gap_learning_results.png")