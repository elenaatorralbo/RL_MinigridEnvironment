import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import warnings

# Minigrid Imports
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Box
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper
from gymnasium.envs.registration import register

# Stable Baselines 3 + Contrib Imports
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==============================================================================
# 1. CUSTOM CNN (Corrección para imágenes pequeñas 7x7)
# ==============================================================================
class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Capa 1: Kernel 2x2 para no perder información en inputs pequeños
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            # Capa 2
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            # Capa 3
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular tamaño de salida automáticamente
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ==============================================================================
# 2. ENTORNO: RUINED TEMPLE (Versión Anti-Trampas y Anti-Bugs)
# ==============================================================================
class RuinedTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 10
        self.grid_h = 10

        mission_space = MissionSpace(mission_func=lambda: "get key and open door")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=500,  # Tiempo suficiente
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        centerY = height // 2

        # --- COLOCACIÓN SEGURA (Evita coordenadas width-1 que son pared) ---

        # 1. Puerta (En x=7)
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(width - 3, centerY, self.door_yellow)

        # 2. Meta (En x=8). IMPORTANTE: width-2 para no chocar con pared derecha
        self.place_obj(Goal(), top=(width - 2, centerY), size=(1, 1))

        # 3. Llave (En el centro)
        self.key_yellow = Key('yellow')
        self.grid.set(width // 2 - 1, centerY, self.key_yellow)

        # 4. Agente (Al inicio)
        self.place_agent(top=(1, 1), size=(2, height - 2))

    def reset(self, *, seed=None, options=None):
        # Reiniciamos las banderas de recompensas
        self.rewarded_key = False
        self.rewarded_door = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # --- LÓGICA DE RECOMPENSAS (ANTI-TRAMPAS) ---

        # 1. CASTIGO POR SOLTAR LA LLAVE (Evita farming)
        if action == self.actions.drop and pre_carrying == self.key_yellow:
            reward -= 10.0
            # print(">>> ¡NO LA SUELTES! <<<")

        # 2. PREMIO ÚNICO POR COGER LLAVE
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            if not self.rewarded_key:
                reward += 10.0
                self.rewarded_key = True
                print(">>> ¡LLAVE COGIDA! <<<")
            else:
                reward += 0.0  # Nada la segunda vez

        # 3. PREMIO ÚNICO POR ABRIR PUERTA
        if not pre_yellow_open and self.door_yellow.is_open:
            if not self.rewarded_door:
                reward += 10.0
                self.rewarded_door = True
                print(">>> ¡PUERTA ABIERTA! <<<")

        # 4. GRAN PREMIO FINAL
        if terminated and reward > 0:
            reward += 50.0

        return obs, reward, terminated, truncated, info


# Registrar entorno con ID nuevo
try:
    register(id='MiniGrid-Kindergarten-Final-v0', entry_point='__main__:RuinedTempleEnv')
except:
    pass


# ==============================================================================
# 3. CONFIGURACIÓN DE ENTRENAMIENTO
# ==============================================================================
def make_env(rank, seed=0):
    def _init():
        env = gym.make('MiniGrid-Kindergarten-Final-v0', render_mode=None)
        env = ImgObsWrapper(env)  # Visión parcial 7x7 (Mejor para generalizar)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":
    # Configuración de rutas
    log_path = "./Training/Logs/"
    save_path = "./Training/Saved_Models/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    warnings.filterwarnings("ignore")

    print(f"--- Iniciando entrenamiento FINAL (Anti-Trampas) ---")

    # 1. Crear entornos paralelos
    num_cpu = 8
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 2. Configurar la red neuronal (CNN + LSTM)
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
    )

    # 3. Inicializar PPO
    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.05,  # Alta exploración inicial
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs
    )

    # 4. Entrenar
    # Con 500k pasos debería dominar este mapa
    model.learn(total_timesteps=20_000)

    # 5. Guardar
    final_model_path = os.path.join(save_path, "PPO_Kindergarten_Final")
    model.save(final_model_path)
    vec_env.close()

    print("--- ¡Entrenamiento terminado! Visualizando... ---")

    # ==============================================================================
    # 4. VISUALIZACIÓN
    # ==============================================================================
    env_test = gym.make('MiniGrid-Kindergarten-Final-v0', render_mode='human')
    env_test = ImgObsWrapper(env_test)

    model = RecurrentPPO.load(final_model_path, env=env_test)

    obs, _ = env_test.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while True:
        # Predecir acción
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        # Ejecutar paso
        obs, reward, terminated, truncated, info = env_test.step(action)
        env_test.render()

        episode_starts = terminated or truncated

        if terminated or truncated:
            print(f"Resultado final del episodio: {reward:.2f}")
            obs, _ = env_test.reset()
            lstm_states = None