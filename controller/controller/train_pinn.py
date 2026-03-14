import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# --- PHYSICS CONSTANTS ---
L = 0.33  # Wheelbase
MU = 1.0  # Increased friction coefficient for better simulator matching
G = 9.81  # Gravity

class F1TENTH_PINN(nn.Module):
    def __init__(self, input_dim):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) 
        )

    def forward(self, x):
        return self.net(x)

def physics_loss(v_pred, delta_total):
    # Ensure delta is within a realistic range to prevent math errors
    delta_abs = torch.abs(delta_total) + 1e-4
    
    # Calculate radius of curvature: R = L / tan(delta)
    radius = L / torch.tan(delta_abs)
    
    # Lateral Acceleration: a = v^2 / R
    lat_accel = (v_pred**2) / radius
    
    # Violation occurs if lateral acceleration exceeds friction limit (MU * G)
    violation = torch.relu(lat_accel - (MU * G))
    return torch.mean(violation)

def main():
    # Detect GPU (CUDA) for your Lenovo LOQ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Clean Data
    csv_path = 'src/controller/controller/training_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Cannot find {csv_path}. Make sure you are running from ~/sim_ws")
        return

    data = pd.read_csv(csv_path)
    # Only keep rows where the car is actually moving
    data = data[data['v_curr'] > 0.1] 

    # Inputs: v_curr, yaw, and lidar_0...lidar_19 (Total 22 columns)
    X = data.drop(columns=['pp_steering', 'pp_speed']).values 
    Y = data[['pp_steering', 'pp_speed']].values            

    # 2. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y, dtype=torch.float32).to(device)
    pp_steering_data = torch.tensor(data['pp_steering'].values, dtype=torch.float32).to(device)

    # 3. Initialize Model with correct input dimension (22)
    input_dim = X_train.shape[1]
    model = F1TENTH_PINN(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Slightly lower learning rate for stability
    mse_crit = nn.MSELoss()

    print(f"🚀 Training on {len(X_train)} samples with {input_dim} inputs...")

    # 4. Training Loop
    for epoch in range(2001): # Increased epochs for better convergence
        optimizer.zero_grad()
        
        predictions = model(X_train)
        delta_corr = predictions[:, 0]
        v_opt = predictions[:, 1]
        
        # Total steering = Pure Pursuit Suggestion + NN Correction
        total_delta = pp_steering_data + delta_corr
        
        # Combined Loss: Data (MSE) + Physics Violation
        loss_data = mse_crit(v_opt, Y_train[:, 1]) + mse_crit(delta_corr, torch.zeros_like(delta_corr))
        loss_phys = physics_loss(v_opt, total_delta)
        
        # Lowered physics weight (0.01) to stop it from fighting the data
        total_loss = loss_data + 0.01 * loss_phys 
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4}: Total Loss {total_loss.item():.4f} | Physics Violation: {loss_phys.item():.4f}")

    # 5. Save Model and Scaler
    model_path = 'src/controller/controller/pinn_model.pth'
    scaler_path = 'src/controller/controller/scaler.pkl'
    
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✅ PINN Training Complete! (Inputs: {input_dim})")
    print(f"💾 Saved model to: {model_path}")
    print(f"💾 Saved scaler to: {scaler_path}")

if __name__ == '__main__':
    main()