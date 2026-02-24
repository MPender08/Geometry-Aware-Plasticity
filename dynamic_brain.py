import torch
import torch.nn as nn
import geoopt

# ==========================================
# 1. THE GLOBAL DIAGNOSTIC HUB (NEW)
# ==========================================
class VIP_MacroController:
    def __init__(self, alpha=0.5, epsilon=0.1, delta=0.01, tau_vip=10.0, ema_alpha=0.1):
        """
        Monitors macroscopic task error to trigger a global hyperbolic plunge.
        alpha: Global gain scalar (intensity of the panic response)
        epsilon: Acceptable error threshold (only inject VIP if error > epsilon)
        delta: Stagnation threshold (inject if dE/dt >= -delta)
        tau_vip: The decay time constant (attention span / physical RC leak rate)
        ema_alpha: Smoothing factor for the error signal (Low-Pass Filter for batch noise)
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.ema_alpha = ema_alpha 
        
        self.decay_factor = torch.exp(torch.tensor(-1.0 / tau_vip))
        
        self.current_gamma_vip = torch.tensor(0.0)
        self.smoothed_error = None
        self.prev_error = None

    def step(self, current_error):
        err_val = current_error.item() if torch.is_tensor(current_error) else current_error
        
        # 1. LOW-PASS FILTER: Smooth the incoming error to ignore batch noise
        if self.smoothed_error is None:
            self.smoothed_error = err_val
        else:
            self.smoothed_error = (self.ema_alpha * err_val) + ((1.0 - self.ema_alpha) * self.smoothed_error)

        if self.prev_error is None:
            self.prev_error = self.smoothed_error
            return self.current_gamma_vip.item()

        # 2. Calculate dE/dt using the SMOOTHED error
        dE_dt = self.smoothed_error - self.prev_error
        
        # 3. Continuous Error Injection (I_VIP)
        stagnating = 1.0 if (dE_dt >= -self.delta) else 0.0
        error_excess = max(0.0, self.smoothed_error - self.epsilon)
        i_vip = self.alpha * error_excess * stagnating
        
        # 4. Metabolic Decay Function
        self.current_gamma_vip = (self.current_gamma_vip * self.decay_factor) + i_vip
        self.current_gamma_vip = torch.clamp(self.current_gamma_vip, 0.0, 1.0)
        
        # Update state for the next step using the smoothed error
        self.prev_error = self.smoothed_error
        
        return self.current_gamma_vip.item()

# ==========================================
# 2. THE LOCAL HARDWARE MODULES (ORIGINAL)
# ==========================================
class HyperbolicLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=1.0)
        self.weight = geoopt.ManifoldParameter(torch.randn(out_features, in_features) * 0.2)
        self.bias = geoopt.ManifoldParameter(torch.zeros(out_features))

    def forward(self, x, current_c):
        # Ensure c is positive and stable
        c = torch.clamp(current_c, min=1e-4, max=5.0)
        temp_manifold = geoopt.PoincareBall(c=c)
        
        # Hyperbolic Transform
        x_hyp = temp_manifold.expmap0(x)
        output_hyp = temp_manifold.mobius_matvec(self.weight, x_hyp)
        output_hyp = temp_manifold.mobius_add(output_hyp, self.bias)
        
        return temp_manifold.logmap0(output_hyp)

class SST_Gate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sense = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.sense(x)).mean()


# ==========================================
# 3. THE UNIFIED ARCHITECTURE (UPDATED)
# ==========================================
class DynamicCurvatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.sst_neuron = SST_Gate(input_dim)
        self.dendrite = HyperbolicLayer(input_dim, hidden_dim)
        
        # REPLACED: standard nn.Linear is now our physical DGAC hardware layer
        self.classifier = GAP_LinearLayer(hidden_dim, output_dim, w_max=2.0, w_min=-2.0)
        
        # Hardware Hysteresis State (The Schmitt Trigger Memory)
        self.is_hyperbolic_locked = torch.tensor(0.0)
        self.gamma_upper = 0.78  # CAH trigger threshold
        self.gamma_lower = 0.40  # Cool-down release threshold

    def forward(self, x, global_gamma_vip=0.0):
        # 1. Bottom-up local calculation (SST Gate)
        local_gamma = self.sst_neuron(x)
        
        # 2. Unified Topological Logic
        gamma_net = torch.clamp(local_gamma + global_gamma_vip, 0.0, 1.0)
        
        # 3. THE SCHMITT TRIGGER (Thermodynamic state memory)
        # We use .item() to detach the logic from the computation graph for the clean 0/1 state
        current_gamma = gamma_net.mean().item()
        if current_gamma >= self.gamma_upper:
            self.is_hyperbolic_locked = torch.tensor(1.0, device=gamma_net.device)
        elif current_gamma <= self.gamma_lower:
            self.is_hyperbolic_locked = torch.tensor(0.0, device=gamma_net.device)
            
        # 4. Hyperbolic Plunge Mapping
        c_mapped = gamma_net * 5.0 
        
        # Approximate instantaneous curvature (kappa) for the learning rule
        # In the CAH, as gamma -> 1, kappa becomes deeply negative
        kappa_current = -c_mapped 
        
        # 5. Route through the dendritic topology
        hidden_state = self.dendrite(x, c_mapped)
        
        # 6. Route through the GAP Crossbar (passing the physical states)
        out = self.classifier(hidden_state, gamma_net.mean(), kappa_current.mean(), self.is_hyperbolic_locked)
        return out, gamma_net.mean()

# ==========================================
# 4: ANALOG CROSSBAR LAYER WITH GAP LEARNING
# ==========================================
class GAP_LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, w_max=2.0, w_min=-2.0):
        """
        w_max and w_min define the physical saturation limits of the memristor oxide.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_max = w_max
        self.w_min = w_min
        
        # Physical memristor conductances (Weights)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize with low, baseline Euclidean conductance
        nn.init.uniform_(self.weight, -0.1, 0.1) 

    def forward(self, x, gamma, kappa, is_hyperbolic_state):
        # Pass the hardware states directly into our custom Autograd function
        return GeometryAwarePlasticity.apply(
            x, self.weight, gamma, kappa, is_hyperbolic_state, self.w_max, self.w_min
        )

# ==========================================
# 5: THE UNIFIED GAP LEARNING RULE (AUTOGRAD)
# ==========================================
class GeometryAwarePlasticity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, gamma, kappa, is_hyperbolic_state, w_max, w_min):
        # 1. Standard Forward Pass (Inference via Kirchhoff's laws)
        output = input.mm(weight.t())
        
        # 2. Save hardware states for the learning phase (Backward Pass)
        ctx.save_for_backward(input, weight, kappa, is_hyperbolic_state)
        ctx.w_max = w_max
        ctx.w_min = w_min
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        This is where the physical "Structural Burn" happens.
        PyTorch optimizers do: W_new = W_old - (lr * grad).
        Therefore:
          - A NEGATIVE grad means POTENTIATION (weight grows).
          - A POSITIVE grad means DEPRESSION (weight shrinks).
        """
        input, weight, kappa, is_hyperbolic_state = ctx.saved_tensors
        w_max = ctx.w_max
        w_min = ctx.w_min
        
        # 1. Calculate Standard STDP / Euclidean Gradient Baseline
        grad_weight_stdp = grad_output.t().mm(input)
        grad_input = grad_output.mm(weight) # Standard gradient routing for previous layers
        
        # 2. The Geometric Multiplier: Phi(gamma)
        alpha = 10.0  # Hardware Amplification Constant (FET Max Current)
        beta = 1.0    # Topological Scaling Factor
        
        # is_hyperbolic_state acts as our Schmitt Trigger (0.0 or 1.0)
        # kappa is negative during the plunge, so exp(-beta * kappa) yields massive amplification
        phi = 1.0 + (alpha * is_hyperbolic_state * torch.exp(-beta * kappa))
        
        # 3. Asymmetric Masking (Potentiation vs. Depression)
        is_potentiation = grad_weight_stdp < 0  # Weight is trying to grow
        is_depression = grad_weight_stdp > 0    # Weight is trying to shrink
        
        # 4. The Physical Governor: Gamma Function Limits
        epsilon = 0.05  # Dampener to protect baseline Euclidean memories
        
        final_grad_weight = torch.zeros_like(grad_weight_stdp)
        
        # --- THE STRUCTURAL BURN (Potentiation) ---
        # Amplified by Phi, bounded by physical (W_max - W) saturation
        bound_pot = torch.clamp(w_max - weight, min=0.0)
        final_grad_weight[is_potentiation] = (
            grad_weight_stdp[is_potentiation] * phi * bound_pot[is_potentiation]
        )
        
        # --- EUCLIDEAN PRUNING (Depression) ---
        # Ignored by Phi (no exponential acceleration), dampened by epsilon, bounded by (W - W_min)
        bound_dep = torch.clamp(weight - w_min, min=0.0)
        final_grad_weight[is_depression] = (
            grad_weight_stdp[is_depression] * epsilon * bound_dep[is_depression]
        )
        
        # Return gradients for all inputs to forward()
        # Grads for gamma, kappa, state, min, max are None because they don't require gradients
        return grad_input, final_grad_weight, None, None, None, None, None