import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# --- PHYSICS CONSTANTS ---
DT = 0.1  # We want to predict 0.1 seconds into the future (approx 5 samples at 50Hz)

class F1TENTH_PINN(nn.Module):
    def __init__(self, input_dim):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) # Outputs: [dx, dy] in local frame
        )

    def forward(self, x):
        return self.net(x)

def physics_consistency_loss(X_raw, Y_pred):
    """
    Ensures predicted dx is consistent with current velocity.
    X_raw contains [v_curr, yaw, lidar...]
    """
    v_curr = X_raw[:, 0]
    # Simple longitudinal physics: dx ≈ v * dt
    expected_dx = v_curr * DT
    return torch.mean((Y_pred[:, 0] - expected_dx)**2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Prepare Data
    csv_path = 'src/controller/controller/training_data.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Error: Cannot find {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    df = df[df['v_curr'] > 0.1].reset_index(drop=True)

    # --- THE DYNAMICS SHIFT ---
    # Calculate global displacement over 5 timesteps
    df['dx_global'] = df['curr_x'].shift(-5) - df['curr_x']
    df['dy_global'] = df['curr_y'].shift(-5) - df['curr_y']
    
    # Convert global displacement to local car frame using current yaw
    # This is critical so the AI learns "Forward/Sideways" movement
    cos_yaw = np.cos(df['yaw'])
    sin_yaw = np.sin(df['yaw'])
    
    df['target_dx'] = df['dx_global'] * cos_yaw + df['dy_global'] * sin_yaw
    df['target_dy'] = -df['dx_global'] * sin_yaw + df['dy_global'] * cos_yaw
    
    df.dropna(inplace=True)

    # Inputs (X): v_curr, yaw, lidar_0...19
    # We ignore the old 'pp' columns as they aren't targets anymore
    lidar_cols = [f'lidar_{i}' for i in range(20)]
    feature_cols = ['v_curr', 'yaw'] + lidar_cols
    X = df[feature_cols].values
    Y = df[['target_dx', 'target_dy']].values

    # 2. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y, dtype=torch.float32).to(device)
    # We keep a non-scaled version for physics loss calculations
    X_raw = torch.tensor(X, dtype=torch.float32).to(device)

    # 3. Initialize
    model = F1TENTH_PINN(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    mse_crit = nn.MSELoss()

    print(f"🚀 Training Dynamics Predictor on {len(X_train)} samples...")

    # 4. Training Loop
    for epoch in range(2501):
        optimizer.zero_grad()
        
        predictions = model(X_train) # Predicted [dx, dy]
        
        # Loss Part 1: Data Accuracy (MSE)
        loss_data = mse_crit(predictions, Y_train)
        
        # Loss Part 2: Physics Consistency
        loss_phys = physics_consistency_loss(X_raw, predictions)
        
        # Total Weighted Loss
        total_loss = loss_data + 0.05 * loss_phys 
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4}: Loss {total_loss.item():.6f} | Phys Viol: {loss_phys.item():.6f}")

    # 5. Save
    model_path = 'src/controller/controller/pinn_model.pth'
    scaler_path = 'src/controller/controller/scaler.pkl'
    
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✅ Dynamics Training Complete! Saved to {model_path}")

if __name__ == '__main__':
    main()