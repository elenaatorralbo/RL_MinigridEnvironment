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
# 1. ENTORNO: THE RUINED TEMPLE (Con Lava, Rocas y Cajas)
# ==========================================
class RuinedTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 42
        self.grid_h = 52
        mission_space = MissionSpace(mission_func=lambda: "avoid lava, explore ruins and find the goal")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=5000,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- DIMENSIONES ---
        HUB_W = 15
        HUB_H = 28
        ROOM_S = 8
        cx, cy = width // 2, height // 2

        hub_x1 = cx - (HUB_W // 2)
        hub_y1 = cy - (HUB_H // 2)
        hub_x2 = hub_x1 + HUB_W
        hub_y2 = hub_y1 + HUB_H

        # 1. CONSTRUIR ESTRUCTURA BÁSICA (PAREDES)
        self.grid.wall_rect(hub_x1, hub_y1, HUB_W, HUB_H)

        # Definición de habitaciones
        rooms = []
        rooms.append({'x': hub_x1, 'y': hub_y1 - ROOM_S + 1, 'pos': 'top'})    # 0
        rooms.append({'x': hub_x2 - ROOM_S, 'y': hub_y1 - ROOM_S + 1, 'pos': 'top'}) # 1
        rooms.append({'x': hub_x2 - 1, 'y': hub_y1, 'pos': 'right'}) # 2
        rooms.append({'x': hub_x2 - 1, 'y': hub_y1 + ROOM_S + 2, 'pos': 'right'}) # 3
        rooms.append({'x': hub_x2 - 1, 'y': hub_y2 - ROOM_S, 'pos': 'right'}) # 4
        rooms.append({'x': hub_x2 - ROOM_S, 'y': hub_y2 - 1, 'pos': 'bottom'}) # 5
        rooms.append({'x': hub_x1, 'y': hub_y2 - 1, 'pos': 'bottom'}) # 6
        rooms.append({'x': hub_x1 - ROOM_S + 1, 'y': hub_y2 - ROOM_S, 'pos': 'left'}) # 7
        rooms.append({'x': hub_x1 - ROOM_S + 1, 'y': hub_y1 + ROOM_S + 2, 'pos': 'left'}) # 8
        rooms.append({'x': hub_x1 - ROOM_S + 1, 'y': hub_y1, 'pos': 'left'}) # 9

        # Construir paredes de habitaciones
        for r in rooms:
            self.grid.wall_rect(r['x'], r['y'], ROOM_S, ROOM_S)

        # ==========================================
        # FASE 0: CÁLCULO DE ZONAS DE SEGURIDAD (GLOBAL)
        # ==========================================
        # Aquí marcamos TODAS las casillas que deben estar limpias (Puertas + Entradas + Salidas)
        # ANTES de poner una sola gota de lava.
        
        global_forbidden = set()

        # 1. Proteger el Spawn (Centro del mapa)
        for i in range(cx - 2, cx + 3):
            for j in range(cy - 2, cy + 3):
                global_forbidden.add((i, j))

        # 2. Calcular posición exacta de puertas y proteger su área (Dentro y Fuera)
        # Esto crea un "túnel seguro" invisible a través de los muros.
        for r in rooms:
            mid_x = r['x'] + ROOM_S // 2
            mid_y = r['y'] + ROOM_S // 2
            
            door_pos = None
            safe_area = []

            if r['pos'] == 'top':
                door_pos = (mid_x, r['y'] + ROOM_S - 1)
                # Protegemos: Puerta, 2 hacia dentro (habitación), 2 hacia fuera (Hub)
                safe_area = [
                    (door_pos[0], door_pos[1]),     # Puerta
                    (door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]-2), # Dentro
                    (door_pos[0], door_pos[1]+1), (door_pos[0], door_pos[1]+2), # Fuera (Hub)
                    # Ensanchar zona (laterales)
                    (door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1]),
                    (door_pos[0]-1, door_pos[1]+1), (door_pos[0]+1, door_pos[1]+1) # Fuera laterales
                ]

            elif r['pos'] == 'bottom':
                door_pos = (mid_x, r['y'])
                safe_area = [
                    (door_pos[0], door_pos[1]),
                    (door_pos[0], door_pos[1]+1), (door_pos[0], door_pos[1]+2), # Dentro
                    (door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]-2), # Fuera
                    (door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1]),
                    (door_pos[0]-1, door_pos[1]-1), (door_pos[0]+1, door_pos[1]-1)
                ]

            elif r['pos'] == 'left':
                door_pos = (r['x'] + ROOM_S - 1, mid_y)
                safe_area = [
                    (door_pos[0], door_pos[1]),
                    (door_pos[0]-1, door_pos[1]), (door_pos[0]-2, door_pos[1]), # Dentro
                    (door_pos[0]+1, door_pos[1]), (door_pos[0]+2, door_pos[1]), # Fuera
                    (door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1),
                    (door_pos[0]+1, door_pos[1]-1), (door_pos[0]+1, door_pos[1]+1)
                ]

            elif r['pos'] == 'right':
                door_pos = (r['x'], mid_y)
                safe_area = [
                    (door_pos[0], door_pos[1]),
                    (door_pos[0]+1, door_pos[1]), (door_pos[0]+2, door_pos[1]), # Dentro
                    (door_pos[0]-1, door_pos[1]), (door_pos[0]-2, door_pos[1]), # Fuera
                    (door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1),
                    (door_pos[0]-1, door_pos[1]-1), (door_pos[0]-1, door_pos[1]+1)
                ]
            
            # Agregamos todo al set global
            for coord in safe_area:
                global_forbidden.add(coord)
            
            # Guardamos la posición de la puerta en el objeto room para usarla luego
            r['door_pos'] = door_pos


        # ==========================================
        # FASE 1: DECORACIÓN INTELIGENTE
        # ==========================================
        
        def decorate_room(x, y, w, h, density=0.1, danger_level=0.0, box_density=0.0, forbidden_coords=None):
            if forbidden_coords is None: forbidden_coords = set()
            
            # A. Protuberancias (Con verificación de colisión)
            side = self._rand_int(0, 4)
            wall_cells = [] 

            # Simular muro
            if side == 0: # Top
                wx = x + self._rand_int(2, w - 2)
                wall_cells = [(wx, y + 1), (wx, y + 2)]
                if not any(c in forbidden_coords for c in wall_cells):
                    self.grid.vert_wall(wx, y + 1, 2)
            elif side == 1: # Right
                wy = y + self._rand_int(2, h - 2)
                wall_cells = [(x + w - 2, wy), (x + w - 3, wy)]
                if not any(c in forbidden_coords for c in wall_cells):
                    self.grid.horz_wall(x + w - 3, wy, 2)
            elif side == 2: # Bottom
                wx = x + self._rand_int(2, w - 2)
                wall_cells = [(wx, y + h - 2), (wx, y + h - 3)]
                if not any(c in forbidden_coords for c in wall_cells):
                    self.grid.vert_wall(wx, y + h - 3, 2)
            elif side == 3: # Left
                wy = y + self._rand_int(2, h - 2)
                wall_cells = [(x + 1, wy), (x + 2, wy)]
                if not any(c in forbidden_coords for c in wall_cells):
                    self.grid.horz_wall(x + 1, wy, 2)

            # B. Relleno interior (Lava/Cajas)
            mid_w = w // 2
            mid_h = h // 2
            
            for i in range(x + 1, x + w - 1):
                for j in range(y + 1, y + h - 1):
                    if self.grid.get(i, j) is not None: continue
                    
                    # CHEQUEO CRÍTICO: ¿Es zona prohibida?
                    if (i, j) in forbidden_coords: continue
                    
                    # Cruz central de seguridad (backup)
                    if i == x + mid_w or j == y + mid_h: continue

                    rng = self._rand_float(0, 1)
                    if rng < danger_level:
                        self.grid.set(i, j, Lava())
                    elif rng < danger_level + box_density:
                        self.grid.set(i, j, Box('grey'))
                    elif rng < danger_level + box_density + density:
                        self.grid.set(i, j, Wall())

        # 1. Decorar Habitaciones (Pasando la máscara global)
        for r in rooms:
            decorate_room(r['x'], r['y'], ROOM_S, ROOM_S, density=0.06, danger_level=0.02, box_density=0.02, forbidden_coords=global_forbidden)

        # 2. Decorar HUB (Pasando la máscara global -> ESTO ARREGLA LA LAVA DELANTE DE LAS PUERTAS)
        decorate_room(hub_x1, hub_y1, HUB_W, HUB_H, density=0.05, danger_level=0.05, box_density=0.02, forbidden_coords=global_forbidden)


        # ==========================================
        # FASE 2: COLOCACIÓN DE OBJETOS
        # ==========================================
        
        def clear_spot(x, y):
             self.grid.set(x, y, None)

        # Room 0 (Inicio)
        r0 = rooms[0]
        # Ya hemos protegido el área en Fase 0, pero hacemos limpieza final por seguridad
        clear_spot(r0['door_pos'][0], r0['door_pos'][1]) 
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(r0['x'] + 1, r0['y'] + 1), size=(ROOM_S - 2, ROOM_S - 2))

        # Room 2 (Puerta Amarilla)
        r2 = rooms[2]
        self.door_yellow = Door('yellow', is_open=False, is_locked=True)
        self.grid.set(r2['door_pos'][0], r2['door_pos'][1], self.door_yellow)
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(r2['x'] + 1, r2['y'] + 1), size=(ROOM_S - 2, ROOM_S - 2))

        # Room 5 (Puerta Roja)
        r4 = rooms[5]
        self.door_red = Door('red', is_open=False, is_locked=True)
        self.grid.set(r4['door_pos'][0], r4['door_pos'][1], self.door_red)
        self.key_blue = Key('blue')
        self.place_obj(self.key_blue, top=(r4['x'] + 1, r4['y'] + 1), size=(ROOM_S - 2, ROOM_S - 2))

        # Room 7 (Puerta Azul -> Meta)
        r6 = rooms[7]
        self.door_blue = Door('blue', is_open=False, is_locked=True)
        self.grid.set(r6['door_pos'][0], r6['door_pos'][1], self.door_blue)
        self.place_obj(Goal(), top=(r6['x'] + 1, r6['y'] + 1), size=(ROOM_S - 2, ROOM_S - 2))

        # Limpieza de puertas extra
        extra_indices = [1, 3, 4, 6, 8, 9]
        for idx in extra_indices:
            r = rooms[idx]
            clear_spot(r['door_pos'][0], r['door_pos'][1])

        # SPAWN FINAL (Limpieza de seguridad extrema)
        for i in range(cx - 1, cx + 2):
            for j in range(cy - 1, cy + 2):
                self.grid.set(i, j, None)
        
        self.place_agent(top=(cx - 1, cy - 1), size=(3, 3))

    # Reset y Step se mantienen igual...
    def reset(self, *, seed=None, options=None):
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False,
            'got_red': False, 'opened_red': False,
            'got_blue': False, 'opened_blue': False
        }
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # (Tu código de step original aquí sin cambios)
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open
        pre_blue_open = self.door_blue.is_open

        obs, reward, terminated, truncated, info = super().step(action)

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
    # Cambiamos el nombre del modelo para reflejar el entrenamiento extenso
    model_path = os.path.join('Training', 'Saved_Models', 'PPO_Ruins_Lava_Tocho')
    os.makedirs(log_path, exist_ok=True)

    print("--- Entrenando THE RUINED TEMPLE (Versión Extensa/Tocho) ---")

    # Aumentamos significativamente los timesteps para un entrenamiento serio
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
        gamma=0.999,  # Súper visión a largo plazo
        ent_coef=0.02,  # Alta exploración necesaria por los obstáculos
        n_steps=4096  # Aumenta el buffer de experiencia recolectada por actualización
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env_train.close()

    print("--- Fin del entrenamiento ---")

    print("--- Visualizando ---")
    env_test = gym.make('MiniGrid-Ruins-v0', render_mode='human')
    env_test = ImgObsWrapper(env_test)

    # El modelo cargará desde la nueva ruta de guardado
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