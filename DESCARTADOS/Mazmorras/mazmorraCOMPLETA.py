import os
import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 1. ARQUITECTURA CNN PERSONALIZADA
# ==========================================
class MinigridCNN(BaseFeaturesExtractor):
    """
    CNN optimizada para imágenes pequeñas (como el grid 7x7 de Minigrid).
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # Capa 1: Filtro 3x3
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Capa 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Capa 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================
# 2. DEFINICIÓN DEL ENTORNO (CORREGIDA)
# ==========================================
class MazmorraSinTrucos(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.size = 10
        # Definimos variables "fijas" para usarlas al reiniciar
        self.start_pos = (1, 1)
        self.start_dir = 0 # 0=Dcha, 1=Abajo, 2=Izq, 3=Arriba
        
        max_steps = 4 * self.size**2 
        
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "llegar a la meta verde"),
            grid_size=self.size,
            max_steps=max_steps,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        # 1. Crear el grid vacío
        self.grid = Grid(width, height)

        # 2. Generar muro exterior
        self.grid.wall_rect(0, 0, width, height)

        # 3. Poner la META
        self.grid.set(width - 2, height - 2, Goal())

        # 4. Poner al AGENTE
        if self.start_pos is not None:
            self.agent_pos = self.start_pos
            # AQUÍ ESTABA EL ERROR: Usamos self.start_dir, no self.agent_dir
            self.agent_dir = self.start_dir 
        else:
            self.place_agent()

        # 5. OBSTÁCULOS
        self.grid.set(4, 4, Wall())
        self.grid.set(4, 5, Wall())
        self.grid.set(4, 6, Wall())
        
        # 6. TRAMPA DE LAVA
        self.grid.set(6, 4, Lava()) 

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Lógica Anti-Hacking
        if truncated:
            reward = -1.0 
        
        if reward == 0 and not terminated:
            reward = -0.01
            
        return obs, reward, terminated, truncated, info

# ==========================================
# 3. CONFIGURACIÓN Y MAIN
# ==========================================
CARPETA_LOGS = "./logs/"
CARPETA_MODELOS = "./modelos_guardados/"
TIMESTEPS_TOTALES = 200000 

def make_env():
    env = MazmorraSinTrucos(render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = Monitor(env, CARPETA_LOGS)
    return env

if __name__ == "__main__":
    os.makedirs(CARPETA_LOGS, exist_ok=True)
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    print("--- Configurando Entorno MazmorraSinTrucos ---")
    
    # Entorno vectorizado
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=CARPETA_MODELOS,
        name_prefix="ppo_mazmorra"
    )

    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    print("Creando modelo PPO con Custom CNN (Optimizado para Minigrid)...")
    model = PPO(
        "CnnPolicy", 
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, 
        verbose=1,
        tensorboard_log=CARPETA_LOGS,
        device="auto" 
    )

    print(f"--- INICIANDO ENTRENAMIENTO ({TIMESTEPS_TOTALES} pasos) ---")
    try:
        model.learn(
            total_timesteps=TIMESTEPS_TOTALES,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente.")
    
    ruta_final = os.path.join(CARPETA_MODELOS, "modelo_FINAL_mazmorra")
    model.save(ruta_final)
    
    print("--- FIN DEL ENTRENAMIENTO ---")
    print(f"Modelo guardado en: {ruta_final}.zip")
    env.close()