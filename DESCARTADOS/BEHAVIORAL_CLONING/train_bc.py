import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class ImitationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImitationNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), # Un poco más grande para Nivel 4
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_cloning():
    filename = 'demonstrations_arcade.pkl' # <--- EL ARCHIVO NUEVO
    print(f"📂 Cargando {filename}...")
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: No encuentro el archivo. Ejecuta record_demo.py primero.")
        return

    # Convertir datos
    observations = np.array([d['obs'] for d in data], dtype=np.float32)
    actions = np.array([d['action'] for d in data], dtype=np.int64)
    
    dataset = TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Configurar modelo
    input_dim = observations.shape[1]
    output_dim = 7 # 0:Left, 1:Right, 2:Fwd, 3:Pickup, 4:Drop, 5:Toggle, 6:Done
    
    model = ImitationNetwork(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🚀 Entrenando clon...")
    
    for epoch in range(30): # 30 épocas suelen sobrar
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_obs, batch_act in dataloader:
            optimizer.zero_grad()
            logits = model(batch_obs)
            loss = criterion(logits, batch_act)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_act).sum().item()
            total += batch_act.size(0)
            
        acc = correct / total * 100
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "bc_lvl4_model.pth")
    print("✅ ¡Modelo entrenado y guardado como 'bc_lvl4_model.pth'!")

if __name__ == "__main__":
    train_cloning()