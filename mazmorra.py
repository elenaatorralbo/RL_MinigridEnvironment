import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Box
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import random

# ==========================================
# 1. ENTORNO: THE RUINED TEMPLE (Con Lava y Rocas)
# ==========================================
class RuinedTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        # Dimensiones para un mapa muy grande
        self.grid_w = 42
        self.grid_h = 44
        
        mission_space = MissionSpace(mission_func=lambda: "avoid lava, explore ruins and find the goal")
        
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=4000, # Aumentamos tiempo porque hay que esquivar lava
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- DIMENSIONES ---
        HUB_W, HUB_H = 15, 20
        ROOM_S = 8 
        cx, cy = width // 2, height // 2
        
        hub_x1 = cx - (HUB_W // 2)
        hub_y1 = cy - (HUB_H // 2)
        hub_x2 = hub_x1 + HUB_W
        hub_y2 = hub_y1 + HUB_H

        # 1. CONSTRUIR HUB CENTRAL
        self.grid.wall_rect(hub_x1, hub_y1, HUB_W, HUB_H)

        # --- FUNCIÓN DECORADORA (ROCAS + LAVA) ---
        def decorate_room(x, y, w, h, density=0.1, danger_level=0.0):
            # A. Protuberancias (Muros irregulares)
            side = self._rand_int(0, 4)
            if side == 0: self.grid.vert_wall(x + self._rand_int(2, w-2), y + 1, 2)
            elif side == 1: self.grid.horz_wall(x + w - 3, y + self._rand_int(2, h-2), 2)
            elif side == 2: self.grid.vert_wall(x + self._rand_int(2, w-2), y + h - 3, 2)
            elif side == 3: self.grid.horz_wall(x + 1, y + self._rand_int(2, h-2), 2)

            # B. Relleno interior (Lava y Rocas)
            for i in range(x + 1, x + w - 1):
                for j in range(y + 1, y + h - 1):
                    # No poner nada si ya hay algo
                    if self.grid.get(i, j) is not None: continue

                    rng = self._rand_float(0, 1)
                    
                    # LAVA (Muerte instantánea)
                    if rng < danger_level:
                        self.grid.set(i, j, Lava())
                    
                    # ROCA (Muro obstáculo)
                    elif rng < danger_level + density:
                        self.grid.set(i, j, Wall())

        # --- CONSTRUIR HABITACIONES ---
        rooms = [] 
        # 2 Arriba
        rooms.append({'x': hub_x1, 'y': hub_y1 - ROOM_S + 1, 'pos': 'top'})
        rooms.append({'x': hub_x2 - ROOM_S, 'y': hub_y1 - ROOM_S + 1, 'pos': 'top'})
        # 2 Derecha
        rooms.append({'x': hub_x2 - 1, 'y': hub_y1, 'pos': 'right'})
        rooms.append({'x': hub_x2 - 1, 'y': hub_y2 - ROOM_S, 'pos': 'right'})
        # 2 Abajo
        rooms.append({'x': hub_x2 - ROOM_S, 'y': hub_y2 - 1, 'pos': 'bottom'})
        rooms.append({'x': hub_x1, 'y': hub_y2 - 1, 'pos': 'bottom'})
        # 2 Izquierda
        rooms.append({'x': hub_x1 - ROOM_S + 1, 'y': hub_y2 - ROOM_S, 'pos': 'left'})
        rooms.append({'x': hub_x1 - ROOM_S + 1, 'y': hub_y1, 'pos': 'left'})

        # Decorar Habitaciones (Poco peligro)
        for r in rooms:
            self.grid.wall_rect(r['x'], r['y'], ROOM_S, ROOM_S)
            decorate_room(r['x'], r['y'], ROOM_S, ROOM_S, density=0.06, danger_level=0.02)

        # Decorar HUB (Mucho peligro: Lava y Ruinas)
        decorate_room(hub_x1, hub_y1, HUB_W, HUB_H, density=0.05, danger_level=0.05)
        
        # --- LIMPIEZA DE SEGURIDAD (CRUCIAL) ---
        # Borramos lava/rocas del centro (spawn) para no morir al nacer
        for i in range(cx - 2, cx + 3):
            for j in range(cy - 2, cy + 3):
                self.grid.set(i, j, None)

        # --- QUEST Y PUERTAS ---
        
        # Room 0 (Inicio - Abierta): Llave Amarilla
        r0 = rooms[0]
        self.grid.set(r0['x']+ROOM_S//2, r0['y']+ROOM_S-1, None) # Entrada
        # Limpiamos entrada para asegurar paso
        self.grid.set(r0['x']+ROOM_S//2, r0['y']+ROOM_S-2, None) 
        
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(r0['x']+1, r0['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # Room 2 (Paso 2): Puerta Amarilla -> Llave Roja
        r2 = rooms[2]
        self.door_yellow = Door('yellow', is_open=False, is_locked=True)
        self.grid.set(r2['x'], r2['y']+ROOM_S//2, self.door_yellow)
        self.grid.set(r2['x']+1, r2['y']+ROOM_S//2, None) # Limpiar tras puerta
        self.grid.set(r2['x']-1, r2['y']+ROOM_S//2, None) # Limpiar ante puerta
        
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(r2['x']+1, r2['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # Room 4 (Paso 3): Puerta Roja -> Llave Azul
        r4 = rooms[4]
        self.door_red = Door('red', is_open=False, is_locked=True)
        self.grid.set(r4['x']+ROOM_S//2, r4['y'], self.door_red)
        self.grid.set(r4['x']+ROOM_S//2, r4['y']+1, None) # Limpiar tras puerta
        self.grid.set(r4['x']+ROOM_S//2, r4['y']-1, None) # Limpiar ante puerta

        self.key_blue = Key('blue')
        self.place_obj(self.key_blue, top=(r4['x']+1, r4['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # Room 6 (Meta): Puerta Azul -> Goal
        r6 = rooms[6]
        self.door_blue = Door('blue', is_open=False, is_locked=True)
        self.grid.set(r6['x']+ROOM_S-1, r6['y']+ROOM_S//2, self.door_blue)
        self.grid.set(r6['x']+ROOM_S-2, r6['y']+ROOM_S//2, None) # Limpiar tras puerta
        self.grid.set(r6['x']+ROOM_S, r6['y']+ROOM_S//2, None)   # Limpiar ante puerta

        self.place_obj(Goal(), top=(r6['x']+1, r6['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # --- ABRIR HUECOS EN SALAS EXTRA ---
        extra_indices = [1, 3, 5, 7]
        for idx in extra_indices:
            r = rooms[idx]
            if r['pos'] == 'top': 
                self.grid.set(r['x']+ROOM_S//2, r['y']+ROOM_S-1, None)
            elif r['pos'] == 'right': 
                self.grid.set(r['x'], r['y']+ROOM_S//2, None)
            elif r['pos'] == 'bottom': 
                self.grid.set(r['x']+ROOM_S//2, r['y'], None)
            elif r['pos'] == 'left': 
                self.grid.set(r['x']+ROOM_S-1, r['y']+ROOM_S//2, None)

        # SPAWN
        self.place_agent(top=(cx-1, cy-1), size=(3, 3))

    def reset(self, *, seed=None, options=None):
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False,
            'got_red': False, 'opened_red': False,
            'got_blue': False, 'opened_blue': False
        }
        return super().reset(seed=seed, options=options)

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open
        pre_blue_open = self.door_blue.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # REWARD SHAPING
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            if not self.rewards_history['got_yellow']:
                reward += 0.5
                self.rewards_history['got_yellow'] = True
        
        if not pre_yellow_open and self.door_yellow.is_open:
            if not self.rewards_history['opened_yellow']:
                reward += 1.0
                self.rewards_history['opened_yellow'] = True

        if pre_carrying != self.key_red and self.carrying == self.key_red:
            if not self.rewards_history['got_red']:
                reward += 1.5
                self.rewards_history['got_red'] = True

        if not pre_red_open and self.door_red.is_open:
            if not self.rewards_history['opened_red']:
                reward += 2.0
                self.rewards_history['opened_red'] = True
        
        if pre_carrying != self.key_blue and self.carrying == self.key_blue:
            if not self.rewards_history['got_blue']:
                reward += 2.5
                self.rewards_history['got_blue'] = True

        if not pre_blue_open and self.door_blue.is_open:
            if not self.rewards_history['opened_blue']:
                reward += 4.0
                self.rewards_history['opened_blue'] = True

        return obs, reward, terminated, truncated, info

# ==========================================
# 2. REGISTRO
# ==========================================
try:
    register(
        id='MiniGrid-Ruins-v0',
        entry_point='__main__:RuinedTempleEnv', 
    )
except:
    pass 

# ==========================================
# 3. RED NEURONAL
# ==========================================
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    log_path = os.path.join('Training', 'Logs')
    model_path = os.path.join('Training', 'Saved_Models', 'PPO_Ruins_Lava')
    os.makedirs(log_path, exist_ok=True)

    print("--- Entrenando THE RUINED TEMPLE (Versión Peligrosa) ---")
    
    total_timesteps = 1000
    
    env_train = gym.make('MiniGrid-Ruins-v0', render_mode=None)
    env_train = ImgObsWrapper(env_train)

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        'CnnPolicy', 
        env_train, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log=log_path,
        learning_rate=0.0003,
        gamma=0.999, # Súper visión a largo plazo
        ent_coef=0.02 # Alta exploración necesaria por los obstáculos
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env_train.close()
    
    print("--- Fin del entrenamiento ---")
    
    print("--- Visualizando ---")
    env_test = gym.make('MiniGrid-Ruins-v0', render_mode='human')
    env_test = ImgObsWrapper(env_test)
    
    model = PPO.load(model_path, env=env_test)
    obs, _ = env_test.reset()
    
    step_counter = 0
    while True:
        step_counter += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        
        if step_counter % 2 == 0:
            env_test.render()
            
        if terminated or truncated:
            print(f"Reward Final: {reward}")
            obs, _ = env_test.reset()