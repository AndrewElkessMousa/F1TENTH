import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# --- CLASS NAME: F1TENTH_PINN (Distinct from standard NN) ---
class F1TENTH_PINN(nn.Module):
    def __init__(self):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [Steering Angle, Speed]
        )
    def forward(self, x):
        return self.net(x)

# --- NEW PHYSICS LOSS CLASS ---
class PINNLoss(nn.Module):
    def __init__(self, wheelbase=0.33, lambda_physics=0.05):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.L = wheelbase
        self.lp = lambda_physics 

    def forward(self, outputs, labels, v_curr):
        # A. Data Loss (Behavioral Cloning)
        loss_steer = self.mse(outputs[:, 0], labels[:, 0])
        loss_speed = self.mse(outputs[:, 1], labels[:, 1])
        loss_data = (loss_steer * 10.0) + (loss_speed * 1.0)

        # B. Physics Loss (Ackermann Constraint)
        # Relationship: Yaw_Rate = (V / L) * tan(delta)
        pred_steer = outputs[:, 0]
        expected_yaw_rate = (v_curr / self.L) * torch.tan(pred_steer)
        true_yaw_rate = (v_curr / self.L) * torch.tan(labels[:, 0])
        
        loss_physics = self.mse(expected_yaw_rate, true_yaw_rate)
        return loss_data + (self.lp * loss_physics)

class F1TenthDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Data file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        feature_cols = ['v_curr', 'err_x', 'err_y', 'err_yaw'] + [f'lidar_{i}' for i in range(20)]
        self.X = df[feature_cols].values
        self.y = df[['steer_label', 'speed_label']].values
        
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        # Explicitly naming the PINN scaler
        scaler_name = 'pinn_scaler.pkl' 
        with open(scaler_name, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 PINN Scaler saved as {scaler_name}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

def main():
    base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
    csv_file = os.path.join(base_path, 'curriculum_training_data.csv') 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  STARTING PHYSICS-INFORMED TRAINING (PINN) ON: {device}")

    dataset = F1TenthDataset(csv_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    
    model = F1TENTH_PINN().to(device)
    criterion = PINNLoss(wheelbase=0.33, lambda_physics=0.05).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    best_val_loss = float('inf')

    for epoch in range(150):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            v_curr = inputs[:, 0]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, v_curr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                v_curr = inputs[:, 0]
                outputs = model(inputs)
                val_loss += criterion(outputs, labels, v_curr).item()
        
        avg_val = val_loss/len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # UNIQUE FILENAME FOR PINN
            torch.save(model.state_dict(), 'pinn_model_weights.pth')
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | PINN Val Loss: {avg_val:.5f}")

    print(f"✅ PINN Training Complete. Best Weight Saved as 'pinn_model_weights.pth'")

if __name__ == "__main__":
    main()