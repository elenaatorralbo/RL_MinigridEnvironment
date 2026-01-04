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
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64), # Capa extra para "pensar" más
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_smart():
    print("🧠 Cargando datos...")
    with open('demonstrations_arcade.pkl', 'rb') as f:
        data = pickle.load(f)

    # Separar datos por tipo de acción
    # 0-3: Moverse, 4: Coger, 5: Abrir
    obs_move = []; act_move = []
    obs_action = []; act_action = []

    for d in data:
        if d['action'] >= 4: # Acciones "Inteligentes" (Coger/Abrir)
            obs_action.append(d['obs'])
            act_action.append(d['action'])
        else: # Acciones "Tontas" (Caminar)
            obs_move.append(d['obs'])
            act_move.append(d['action'])

    print(f"📊 Estadísticas originales: {len(obs_move)} pasos vs {len(obs_action)} interacciones.")
    
    # --- TRUCO: MULTIPLICAR LOS EJEMPLOS IMPORTANTES ---
    # Repetimos las interacciones 30 veces para que la IA no las ignore
    obs_action = obs_action * 30
    act_action = act_action * 30
    print(f"📈 Estadísticas trucadas: {len(obs_move)} pasos vs {len(obs_action)} interacciones.")

    # Juntar todo de nuevo
    all_obs = np.array(obs_move + obs_action, dtype=np.float32)
    all_acts = np.array(act_move + act_action, dtype=np.int64)

    dataset = TensorDataset(torch.from_numpy(all_obs), torch.from_numpy(all_acts))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Configurar Red
    input_dim = all_obs.shape[1]
    output_dim = 6 # Arcade (0-5)
    
    model = ImitationNetwork(input_dim, output_dim)
    
    # Usamos pesos también por si acaso
    counts = np.bincount(all_acts, minlength=6)
    weights = len(all_acts) / (len(counts) * (counts + 1e-5))
    class_weights = torch.FloatTensor(weights)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Learning rate más bajo para afinar
    
    print("🚀 Entrenando clon inteligente...")
    
    for epoch in range(40):
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
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_act).sum().item()
            total += batch_act.size(0)
            
        if epoch % 5 == 0:
            acc = correct / total * 100
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.2f} | Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "bc_smart_model.pth")
    print("✅ Modelo 'bc_smart_model.pth' guardado.")

if __name__ == "__main__":
    train_smart()