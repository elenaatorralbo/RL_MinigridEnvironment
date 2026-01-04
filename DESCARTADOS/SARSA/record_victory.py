import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment import TwoRoomEnv  # Importamos tu entorno

# --- IMPORTANTE: DEFINICIÓN DE LA RED ---
# Debemos incluir esta clase exactamente igual que en el entrenamiento
# para que SB3 pueda cargar los pesos correctamente.
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = 1 
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float() / 255.0))

def main():
    # 1. RUTA DEL MODELO
    # Busca en tu carpeta modelos_2rooms el archivo con 20000 pasos
    model_path = "./modelos_2rooms/rl_2room_20000_steps.zip" 
    
    # Si no encuentras el de 20k, prueba con el final:
    # model_path = "agente_maestro_2rooms.zip"

    print(f"Cargando modelo desde: {model_path}")

    # 2. CREAR ENTORNO VISUAL
    # render_mode='human' abrirá la ventana de Pygame
    env = TwoRoomEnv(
        render_mode='human', 
        room_size=8,
        agent_view_size=7
    )

    # 3. CARGAR AGENTE
    try:
        model = RecurrentPPO.load(model_path, custom_objects={
            "features_extractor_class": MinigridFeaturesExtractor
        })
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {model_path}")
        print("Asegúrate de haber entrenado al menos 20,000 pasos.")
        return

    # 4. BUCLE DE PRUEBA
    episodes = 5
    
    for ep in range(episodes):
        obs, info = env.reset()
        
        # Inicializar estados de la memoria LSTM (importante para RecurrentPPO)
        lstm_states = None
        num_envs = 1
        # El inicio del episodio marca cuándo resetear la memoria LSTM interna
        episode_starts = np.ones((num_envs,), dtype=bool)
        
        terminated = False
        truncated = False
        total_reward = 0
        step_counter = 0
        
        print(f"\n--- Episodio {ep + 1} ---")
        
        while not (terminated or truncated):
            # El modelo predice la acción y el nuevo estado de memoria (lstm_states)
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts
            )
            
            # Ejecutar paso en el entorno
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_counter += 1
            
            # Actualizar flag de inicio de episodio (ya no es el inicio)
            episode_starts = np.zeros((num_envs,), dtype=bool)
            
            # Renderizar (env.render() se llama internamente con render_mode='human')
            env.render()
            
            # Pequeña pausa para que el ojo humano pueda seguir el movimiento
            time.sleep(0.1) 

        print(f"Fin episodio. Pasos: {step_counter}, Recompensa Total: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()