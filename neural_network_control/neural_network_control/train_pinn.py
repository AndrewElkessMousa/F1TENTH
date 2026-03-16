import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# 1. ARCHITECTURE 
class F1TENTH_PINN(nn.Module):
    def __init__(self):
        super(F1TENTH_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# 2. DATASET LOADER
class F1TenthDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Data file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples from {csv_path}")
        
        feature_cols = ['v_curr', 'err_x', 'err_y', 'err_yaw'] + [f'lidar_{i}' for i in range(20)]
        self.X = df[feature_cols].values
        self.y = df[['steer_label', 'speed_label']].values
        
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        # Save scaler in the same directory as the script for easy access
        scaler_name = 'scaler_v2.pkl'
        with open(scaler_name, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Scaler saved as {scaler_name}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

# 3. MAIN TRAINING FUNCTION
def main(args=None): # Added args=None for ROS 2 compatibility
    # --- FIXED PATH LOGIC ---
    base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
    csv_file = os.path.join(base_path, 'curriculum_training_data.csv') 
    
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Training on: {device}")
    print(f"📂 Target File: {csv_file}")

    # Load Data
    try:
        dataset = F1TenthDataset(csv_file)
    except FileNotFoundError as e:
        print(e)
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = F1TENTH_PINN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')

    print("🚀 Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_steer = criterion(outputs[:, 0], labels[:, 0])
            loss_speed = criterion(outputs[:, 1], labels[:, 1])
            loss = (loss_steer * 10.0) + (loss_speed * 1.0) 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item()
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save the model in the current working directory
            torch.save(model.state_dict(), 'pinn_model_v2.pth')
            if epoch > 5: print(f"✨ New Best Val Loss! Model saved at epoch {epoch+1}")

    print(f"✅ Training Complete. Best Validation Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    main()