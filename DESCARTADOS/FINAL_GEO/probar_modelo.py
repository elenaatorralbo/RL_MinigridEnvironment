import gymnasium as gym
import torch as th
import torch.nn as nn
import time
import os
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment import SurvivalCorridorEnv  # Asegúrate de que environment.py esté en la carpeta

# --- 1. DEFINICIÓN DEL EXTRACTOR (NECESARIO PARA CARGAR EL MODELO) ---
# SB3 necesita tener la clase definida exactamente igual que cuando entrenaste
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
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
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float() / 10.0))

def probar_agente():
    # --- CONFIGURACIÓN ---
    # Pon aquí el nombre exacto de tu archivo guardado (sin el .zip si quieres)
    MODELO_PATH = "checkpoints_inverso/rl_inverso_50000_steps.zip" 
    
    # Comprobamos si existe el archivo
    if not os.path.exists(MODELO_PATH + ".zip") and not os.path.exists(MODELO_PATH):
        print(f"❌ ERROR: No encuentro el modelo '{MODELO_PATH}'. Verifica el nombre.")
        return

    while True:
        print("\n" + "="*40)
        print("🤖 MODO DE PRUEBA DE AGENTE")
        print("="*40)
        try:
            user_input = input("Introduce el número de habitación para empezar (0-24) o 'q' para salir: ")
            if user_input.lower() == 'q':
                break
            
            start_room = int(user_input)
            if start_room < 0 or start_room > 24:
                print("⚠️ Por favor, elige un número entre 0 y 24.")
                continue

        except ValueError:
            print("⚠️ Entrada no válida. Introduce un número.")
            continue

        print(f"\n🎬 Cargando entorno en Habitación {start_room}...")

        # 1. Crear entorno con renderizado HUMANO
        env = SurvivalCorridorEnv(
            render_mode='human',  # <--- ESTO ACTIVA LA VENTANA VISUAL
            num_rooms=25, 
            agent_view_size=9, 
            agent_start_room=start_room
        )
        env = ImgObsWrapper(env)

        # 2. Cargar el modelo
        print("🧠 Cargando cerebro del agente...")
        try:
            model = RecurrentPPO.load(MODELO_PATH, env=env)
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            env.close()
            break

        # 3. Bucle de juego
        obs, _ = env.reset()
        
        # Inicializar estados LSTM (Memoria a cero)
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        
        total_reward = 0
        steps = 0
        done = False
        
        print("🚀 ¡Acción! (Pulsa Ctrl+C en la consola para cancelar el episodio actual)")

        try:
            while not done:
                # El modelo predice la acción usando la observación y su memoria (lstm_states)
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=False # Importante: True para ver su mejor comportamiento, False para ver variedad
                )
                action = int(action)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_starts = done
                
                total_reward += reward
                steps += 1
                
                # Renderizado automático por 'render_mode="human"'
                # Control de velocidad (FPS)
                time.sleep(0.05) # Cambia esto para ir más rápido (0.01) o más lento (0.2)

            print("-" * 30)
            print(f"🏁 Episodio terminado.")
            print(f"💰 Recompensa Total: {total_reward:.2f}")
            print(f"👣 Pasos dados: {steps}")
            if total_reward > 40:
                print("🏆 Resultado: ¡VICTORIA!")
            else:
                print("💀 Resultado: Muerte o Tiempo Agotado.")
        
        except KeyboardInterrupt:
            print("\n🛑 Episodio interrumpido por el usuario.")
        
        finally:
            env.close()
            print("Cerrando ventana...")

if __name__ == "__main__":
    import numpy as np # Necesario para los estados LSTM
    probar_agente()