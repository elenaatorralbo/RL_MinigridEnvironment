import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import pygame
import time
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper

# Importar tu entorno
from custom_env import CurriculumTempleEnv

# Registro y Limpieza
if 'MiniGrid-Curriculum-Arcade-v0' in gym.envs.registry:
    del gym.envs.registry['MiniGrid-Curriculum-Arcade-v0']
gym.register(id='MiniGrid-Curriculum-Arcade-v0', entry_point='custom_env:CurriculumTempleEnv')

# --- ARQUITECTURA "SMART" (DEBE SER IDÉNTICA A TRAIN_SMART.PY) ---
class ImitationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImitationNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64), # <--- ¡ESTA CAPA FALTABA!
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def enjoy():
    print("🎮 Cargando entorno...")
    env = gym.make('MiniGrid-Curriculum-Arcade-v0', render_mode='human')
    env.unwrapped.set_level(4)

    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)

    obs, _ = env.reset()
    
    # --- CORRECCIÓN DEL TAMAÑO DE ENTRADA ---
    # El modelo guardado espera 3771 (según tu error).
    # Si el entorno actual da 3363, tenemos un problema de versión.
    # Vamos a forzar la dimensión del modelo para que cargue.
    
    current_input_dim = obs.shape[0]
    model_expected_dim = 3771 # <--- Sacado de tu mensaje de error
    output_dim = 6 
    
    model_name = "bc_smart_model.pth"
    print(f"🧠 Cargando cerebro: {model_name}...")
    print(f"   (Entorno da: {current_input_dim}, Modelo pide: {model_expected_dim})")

    # Creamos la red con lo que PIDE EL ARCHIVO .PTH
    model = ImitationNetwork(model_expected_dim, output_dim)
    
    try:
        model.load_state_dict(torch.load(model_name))
        print("✅ ¡Cerebro cargado!")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        print("💡 CONSEJO: Si esto falla, es mejor volver a grabar (record_demo.py) y entrenar de nuevo.")
        return

    model.eval()

    print("\n🍿 PULSA LA PANTALLA... ¡Acción!")
    time.sleep(1)
    
    action_names = {0:"⬅️", 1:"➡️", 2:"⬆️", 3:"⬇️", 4:"✋ COGER", 5:"🔑 ABRIR"}

    for episode in range(5):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"\n--- Episodio {episode + 1} ---")

        while not (terminated or truncated):
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: env.close(); return

            # --- TRUCO DE COMPATIBILIDAD ---
            # Si el entorno da menos datos de los que el modelo quiere, rellenamos con ceros
            if obs.shape[0] != model_expected_dim:
                diff = model_expected_dim - obs.shape[0]
                # Rellenamos con ceros (padding) para que no explote
                obs_fixed = np.pad(obs, (0, diff), 'constant')
                obs_tensor = torch.tensor(obs_fixed, dtype=torch.float32).unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1)
                confidence = probs[0][action].item() * 100

            act_str = action_names.get(action, "?")
            print(f"Paso {step}: {act_str} ({confidence:.1f}%)")

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            
            if action == 4 and reward > 1.0: print("✨ ¡HA DECIDIDO COGER LA LLAVE!")
            if action == 5 and reward > 1.0: print("🚪 ¡HA DECIDIDO ABRIR LA PUERTA!")
            
            pygame.time.wait(100)

        if total_reward >= 50: print("✅ ¡VICTORIA!")
        else: print("💀 Se atascó.")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    enjoy()