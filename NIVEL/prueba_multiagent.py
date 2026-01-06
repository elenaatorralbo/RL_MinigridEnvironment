import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import warnings

# Minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Box
from minigrid.minigrid_env import MiniGridEnv
# IMPORTANTE: Añadido FullyObsWrapper
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.envs.registration import register

# Stable Baselines 3 + Contrib
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==============================================================================
# 1. CUSTOM CNN (Soporta tanto visión parcial como total)
# ==============================================================================
class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ==============================================================================
# 2. ENTORNO: THE RUINED TEMPLE (32x32)
# ==============================================================================
class RuinedTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 32
        self.grid_h = 32

        mission_space = MissionSpace(mission_func=lambda: "traverse the ruins, find keys and reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2000,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        split_idx_x = width // 2
        split_idx_y1 = height // 3
        split_idx_y2 = (height // 3) * 2

        self.grid.horz_wall(0, split_idx_y1, width)
        self.grid.horz_wall(0, split_idx_y2, width)

        # -- SECCIÓN 1 --
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(split_idx_x, split_idx_y1, self.door_yellow)
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(1, 1), size=(width - 2, split_idx_y1 - 2))

        # -- SECCIÓN 2 --
        self.door_red = Door('red', is_locked=True)
        self.grid.set(split_idx_x - 5, split_idx_y2, self.door_red)
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(1, split_idx_y1 + 1), size=(width - 2, (split_idx_y2 - split_idx_y1) - 2))

        # -- SECCIÓN 3 --
        self.place_obj(Goal(), top=(1, split_idx_y2 + 1), size=(width - 2, (height - split_idx_y2) - 2))

        self._add_obstacles(density=0.05)
        self.place_agent(top=(1, 1), size=(width // 2, split_idx_y1 - 2))

    def _add_obstacles(self, density):
        for i in range(1, self.width - 1):
            for j in range(1, self.height - 1):
                if self.grid.get(i, j) is None:
                    rng = self._rand_float(0, 1)
                    if rng < density:
                        self.grid.set(i, j, Lava())
                    elif rng < density + 0.02:
                        self.grid.set(i, j, Box('grey'))

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # Reward Shaping
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            reward += 0.5
        if not pre_yellow_open and self.door_yellow.is_open:
            reward += 1.0
        if pre_carrying != self.key_red and self.carrying == self.key_red:
            reward += 1.5
        if not pre_red_open and self.door_red.is_open:
            reward += 2.0

        reward -= 0.001
        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-Ruins-Pro-v0', entry_point='__main__:RuinedTempleEnv')
except:
    pass


# ==============================================================================
# 3. UTILIDADES DE ENTRENAMIENTO (MODIFICADO SOLUCIÓN 1)
# ==============================================================================

def make_env(rank, seed=0):
    def _init():
        env = gym.make('MiniGrid-Ruins-Pro-v0', render_mode=None)
        # <<< SOLUCIÓN 1: FullyObsWrapper >>>
        # Esto hace que el agente vea TODO el mapa (32x32), no solo 7x7.
        env = FullyObsWrapper(env)

        env = ImgObsWrapper(env)  # Convierte obs a imagen
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ==============================================================================
# 4. MAIN
# ==============================================================================
if __name__ == "__main__":
    log_path = "./Training/Logs/"
    save_path = "./Training/Saved_Models/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    warnings.filterwarnings("ignore")

    # --- PARTE A: ENTRENAMIENTO ---
    num_cpu = 8
    print(f"--- Iniciando entrenamiento (Modo Dios + Alta Entropía) ---")

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=256),  # Aumentado un poco para procesar mapa completo
        lstm_hidden_size=256,
        enable_critic_lstm=False,
    )

    # <<< SOLUCIÓN 3: AJUSTE DE HIPERPARÁMETROS >>>
    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0001,  # Bajado para más estabilidad
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,  # Subido (Alta exploración)
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_cpu,
        save_path=save_path,
        name_prefix="ppo_ruins_godmode"
    )

    # Entrenar
    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    final_model_path = os.path.join(save_path, "PPO_Ruins_Final_GodMode")
    model.save(final_model_path)
    vec_env.close()
    print("--- Entrenamiento Finalizado ---")

    # --- PARTE B: VISUALIZACIÓN ---
    print("--- Visualizando el resultado ---")

    env_test = gym.make('MiniGrid-Ruins-Pro-v0', render_mode='human')
    # Recordar aplicar los mismos wrappers en test
    env_test = FullyObsWrapper(env_test)  # <<< IMPORTANTE EN TEST TAMBIÉN
    env_test = ImgObsWrapper(env_test)

    model = RecurrentPPO.load(final_model_path, env=env_test)

    obs, _ = env_test.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env_test.step(action)
        env_test.render()

        episode_starts = terminated or truncated

        if terminated or truncated:
            print(f"Episodio terminado. Recompensa: {reward:.2f}")
            obs, _ = env_test.reset()
            lstm_states = None
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import warnings

# Minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Box
from minigrid.minigrid_env import MiniGridEnv
# IMPORTANTE: Añadido FullyObsWrapper
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.envs.registration import register

# Stable Baselines 3 + Contrib
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==============================================================================
# 1. CUSTOM CNN (Soporta tanto visión parcial como total)
# ==============================================================================
class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ==============================================================================
# 2. ENTORNO: THE RUINED TEMPLE (32x32)
# ==============================================================================
class RuinedTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 32
        self.grid_h = 32

        mission_space = MissionSpace(mission_func=lambda: "traverse the ruins, find keys and reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2000,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        split_idx_x = width // 2
        split_idx_y1 = height // 3
        split_idx_y2 = (height // 3) * 2

        self.grid.horz_wall(0, split_idx_y1, width)
        self.grid.horz_wall(0, split_idx_y2, width)

        # -- SECCIÓN 1 --
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(split_idx_x, split_idx_y1, self.door_yellow)
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(1, 1), size=(width - 2, split_idx_y1 - 2))

        # -- SECCIÓN 2 --
        self.door_red = Door('red', is_locked=True)
        self.grid.set(split_idx_x - 5, split_idx_y2, self.door_red)
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(1, split_idx_y1 + 1), size=(width - 2, (split_idx_y2 - split_idx_y1) - 2))

        # -- SECCIÓN 3 --
        self.place_obj(Goal(), top=(1, split_idx_y2 + 1), size=(width - 2, (height - split_idx_y2) - 2))

        self._add_obstacles(density=0.05)
        self.place_agent(top=(1, 1), size=(width // 2, split_idx_y1 - 2))

    def _add_obstacles(self, density):
        for i in range(1, self.width - 1):
            for j in range(1, self.height - 1):
                if self.grid.get(i, j) is None:
                    rng = self._rand_float(0, 1)
                    if rng < density:
                        self.grid.set(i, j, Lava())
                    elif rng < density + 0.02:
                        self.grid.set(i, j, Box('grey'))

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # Reward Shaping
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            reward += 0.5
        if not pre_yellow_open and self.door_yellow.is_open:
            reward += 1.0
        if pre_carrying != self.key_red and self.carrying == self.key_red:
            reward += 1.5
        if not pre_red_open and self.door_red.is_open:
            reward += 2.0

        reward -= 0.001
        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-Ruins-Pro-v0', entry_point='__main__:RuinedTempleEnv')
except:
    pass


# ==============================================================================
# 3. UTILIDADES DE ENTRENAMIENTO (MODIFICADO SOLUCIÓN 1)
# ==============================================================================

def make_env(rank, seed=0):
    def _init():
        env = gym.make('MiniGrid-Ruins-Pro-v0', render_mode=None)
        # <<< SOLUCIÓN 1: FullyObsWrapper >>>
        # Esto hace que el agente vea TODO el mapa (32x32), no solo 7x7.
        env = FullyObsWrapper(env)

        env = ImgObsWrapper(env)  # Convierte obs a imagen
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ==============================================================================
# 4. MAIN
# ==============================================================================
if __name__ == "__main__":
    log_path = "./Training/Logs/"
    save_path = "./Training/Saved_Models/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    warnings.filterwarnings("ignore")

    # --- PARTE A: ENTRENAMIENTO ---
    num_cpu = 8
    print(f"--- Iniciando entrenamiento (Modo Dios + Alta Entropía) ---")

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=256),  # Aumentado un poco para procesar mapa completo
        lstm_hidden_size=256,
        enable_critic_lstm=False,
    )

    # <<< SOLUCIÓN 3: AJUSTE DE HIPERPARÁMETROS >>>
    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0001,  # Bajado para más estabilidad
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,  # Subido (Alta exploración)
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_cpu,
        save_path=save_path,
        name_prefix="ppo_ruins_godmode"
    )

    # Entrenar
    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    final_model_path = os.path.join(save_path, "PPO_Ruins_Final_GodMode")
    model.save(final_model_path)
    vec_env.close()
    print("--- Entrenamiento Finalizado ---")

    # --- PARTE B: VISUALIZACIÓN ---
    print("--- Visualizando el resultado ---")

    env_test = gym.make('MiniGrid-Ruins-Pro-v0', render_mode='human')
    # Recordar aplicar los mismos wrappers en test
    env_test = FullyObsWrapper(env_test)  # <<< IMPORTANTE EN TEST TAMBIÉN
    env_test = ImgObsWrapper(env_test)

    model = RecurrentPPO.load(final_model_path, env=env_test)

    obs, _ = env_test.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env_test.step(action)
        env_test.render()

        episode_starts = terminated or truncated

        if terminated or truncated:
            print(f"Episodio terminado. Recompensa: {reward:.2f}")
            obs, _ = env_test.reset()
            lstm_states = None