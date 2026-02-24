import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import geoopt

# Import our updated architecture and the new Diagnostic Hub
from dynamic_brain import DynamicCurvatureNet, VIP_MacroController

# --- SEED CONTROL ---
USE_LOCKED_SEED = True
LOCKED_SEED = 137

if USE_LOCKED_SEED:
    current_seed = LOCKED_SEED
else:
    current_seed = random.randint(1000, 99999)

print("\n" + "="*60)
print(f"SEED: {current_seed}")
print("="*60 + "\n")

torch.manual_seed(current_seed)

# --- Simulation Logic ---
x_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y_data = torch.tensor([[0.], [1.], [1.], [0.]])
mse = nn.MSELoss()

def run_stamina_test(label, tax_rate, use_vip=False):
    print(f"Running Stamina Test: {label} (VIP Override: {use_vip})")
    
    model = DynamicCurvatureNet(input_dim=2, hidden_dim=2, output_dim=1)
    
    # Bias the brain to start slightly Hyperbolic
    nn.init.constant_(model.sst_neuron.sense[2].bias, 2.0)
    
    optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=0.01)
    
    # Track the geometric and physical state every 10 epochs
    history = {"gamma": [], "error": [], "vip": [], "max_weights": [], "mean_weights": []}
    
    vip_hub = VIP_MacroController(alpha=0.8, epsilon=0.15, delta=0.005, tau_vip=30.0) if use_vip else None
    current_vip_signal = 0.0
    
    for epoch in range(1500):
        optimizer.zero_grad()
        
        # Forward pass returning both prediction and the geometric state
        pred, gamma_net = model(x_data, global_gamma_vip=current_vip_signal)
        
        task_loss = mse(pred, y_data)
        tax_loss = tax_rate * (gamma_net ** 2)
        total_loss = (task_loss * 20.0) + tax_loss
        
        total_loss.backward()
        optimizer.step()
        
        # --- NEW: The Physical Hardware Clamp ---
        # Enforce the absolute physical capacity of the memristor oxide
        with torch.no_grad():
            model.classifier.weight.data.clamp_(min=-2.0, max=2.0)
        
        if use_vip:
            current_vip_signal = vip_hub.step(task_loss)
        
        # Log data every 10 epochs
        if epoch % 10 == 0:
            history["gamma"].append(gamma_net.item())
            history["error"].append(task_loss.item())
            history["vip"].append(current_vip_signal)
            
            # Track the physical hardware state
            current_w = model.classifier.weight.data.abs()
            history["max_weights"].append(current_w.max().item())
            history["mean_weights"].append(current_w.mean().item())
            
    return history

# --- Run Experiments ---
# We run a standard healthy model, and a healthy model with the VIP override active
healthy_standard = run_stamina_test("Healthy (Local Only)", tax_rate=0.001, use_vip=False)
healthy_vip = run_stamina_test("Healthy (VIP Attention)", tax_rate=0.001, use_vip=True)

# --- Advanced Visualization ---

plt.figure(figsize=(18, 6))
epochs_x = np.arange(0, 1500, 10)

# Plot A: Cognitive Performance (Error)
plt.subplot(1, 3, 1)
plt.plot(epochs_x, healthy_standard["error"], 'g--', lw=2, alpha=0.6, label="Local Only")
plt.plot(epochs_x, healthy_vip["error"], 'b', lw=3, label="VIP Override Active")
plt.axhline(0.25, color='k', linestyle=':', label="Random Guess Limit")
plt.ylim(-0.02, 0.52)
plt.title("Cognitive Performance (XOR Task)")
plt.xlabel("Epochs")
plt.ylabel("MSE Error")
plt.legend()
plt.grid(alpha=0.3)

# Plot B: Dendritic Curvature (Gamma)
plt.subplot(1, 3, 2)
plt.plot(epochs_x, healthy_standard["gamma"], 'g--', lw=2, alpha=0.6, label="Local Only")
plt.plot(epochs_x, healthy_vip["gamma"], 'b', lw=3, label="VIP Override Active")
plt.title("Network Curvature ($\\gamma_{net}$)")
plt.xlabel("Epochs")
plt.ylabel("Geometric State (0=Flat, 1=Hyperbolic)")
plt.legend()
plt.grid(alpha=0.3)

# Plot C: The Physical Geometry (Memristor Conductance Divergence)
plt.subplot(1, 3, 3) 
plt.plot(epochs_x, healthy_vip["mean_weights"], 'k:', lw=2, label="Mean Conductance (Euclidean Baseline)")
plt.plot(epochs_x, healthy_vip["max_weights"], 'r', lw=2, label="Max Conductance (Geodesic Highway)")

# Overlay the Gamma spikes to show the exact trigger moments
gamma_scaled = np.array(healthy_vip["gamma"]) * 2.0 
plt.fill_between(epochs_x, 0, gamma_scaled, color='blue', alpha=0.1, label="VIP Trigger (Hyperbolic Plunge)")

plt.axhline(2.0, color='red', linestyle='--', label="W_max (Physical Saturation)")
plt.title("Memristor Conductance Divergence (GAP)")
plt.xlabel("Epochs")
plt.ylabel("Absolute Weight Value")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()