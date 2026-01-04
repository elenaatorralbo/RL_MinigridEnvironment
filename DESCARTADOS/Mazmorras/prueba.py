import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os

# Imports de MiniGrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Box
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper

# Imports de Gymnasium y SB3
from gymnasium.envs.registration import register
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results
from sb3_contrib import RecurrentPPO

# ==========================================
# 1. ENTORNO: THE RUINED TEMPLE (Curriculum)
# ==========================================
class RuinedTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 42
        self.grid_h = 52
        
        # Estado del Curriculum
        self.curriculum_stage = 0 

        mission_space = MissionSpace(mission_func=lambda: "avoid lava, explore ruins and find the goal")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2000, # Reducido un poco para facilitar convergencia inicial
            render_mode=render_mode
        )

    def set_curriculum_stage(self, stage):
        self.curriculum_stage = stage

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- DIMENSIONES ---
        HUB_W = 15; HUB_H = 28; ROOM_S = 8
        cx, cy = width // 2, height // 2
        hub_x1 = cx - (HUB_W // 2); hub_y1 = cy - (HUB_H // 2)
        hub_x2 = hub_x1 + HUB_W; hub_y2 = hub_y1 + HUB_H

        # Estructura Hub
        self.grid.wall_rect(hub_x1, hub_y1, HUB_W, HUB_H)

        # Definición de Habitaciones
        rooms = [
            {'x': hub_x1, 'y': hub_y1 - ROOM_S + 1, 'pos': 'top'},    # 0 
            {'x': hub_x2 - ROOM_S, 'y': hub_y1 - ROOM_S + 1, 'pos': 'top'}, 
            {'x': hub_x2 - 1, 'y': hub_y1, 'pos': 'right'},           # 2
            {'x': hub_x2 - 1, 'y': hub_y1 + ROOM_S + 2, 'pos': 'right'}, 
            {'x': hub_x2 - 1, 'y': hub_y2 - ROOM_S, 'pos': 'right'}, 
            {'x': hub_x2 - ROOM_S, 'y': hub_y2 - 1, 'pos': 'bottom'}, # 5
            {'x': hub_x1, 'y': hub_y2 - 1, 'pos': 'bottom'}, 
            {'x': hub_x1 - ROOM_S + 1, 'y': hub_y2 - ROOM_S, 'pos': 'left'}, # 7 (Goal)
            {'x': hub_x1 - ROOM_S + 1, 'y': hub_y1 + ROOM_S + 2, 'pos': 'left'}, 
            {'x': hub_x1 - ROOM_S + 1, 'y': hub_y1, 'pos': 'left'}
        ]

        # Construir paredes
        for r in rooms:
            self.grid.wall_rect(r['x'], r['y'], ROOM_S, ROOM_S)

        # Calcular posiciones de puertas
        for r in rooms:
            mid_x = r['x'] + ROOM_S // 2
            mid_y = r['y'] + ROOM_S // 2
            if r['pos'] == 'top': r['door_pos'] = (mid_x, r['y'] + ROOM_S - 1)
            elif r['pos'] == 'bottom': r['door_pos'] = (mid_x, r['y'])
            elif r['pos'] == 'left': r['door_pos'] = (r['x'] + ROOM_S - 1, mid_y)
            elif r['pos'] == 'right': r['door_pos'] = (r['x'], mid_y)
            self.grid.set(r['door_pos'][0], r['door_pos'][1], None)

        # --- DECORACIÓN SEGURA ---
        spawn_room_idx = -1
        if self.curriculum_stage == 0: spawn_room_idx = 7  # Goal Room
        elif self.curriculum_stage == 1: spawn_room_idx = 5 # Blue Key Room
        elif self.curriculum_stage == 2: spawn_room_idx = 2 # Red Key Room
        elif self.curriculum_stage == 3: spawn_room_idx = 0 # Yellow Key Room
        
        safe_rects = []
        for r in rooms:
            dp = r['door_pos']
            safe_rects.append((dp[0]-1, dp[1]-1, dp[0]+1, dp[1]+1))

        if spawn_room_idx != -1:
            r = rooms[spawn_room_idx]
            safe_rects.append((r['x']+1, r['y']+1, r['x']+ROOM_S-2, r['y']+ROOM_S-2))
        
        if self.curriculum_stage >= 4:
             safe_rects.append((cx-2, cy-2, cx+2, cy+2))

        def is_safe(tx, ty):
            for (x1, y1, x2, y2) in safe_rects:
                if x1 <= tx <= x2 and y1 <= ty <= y2:
                    return True
            return False

        # Llenar con Lava
        for i in range(1, width-1):
            for j in range(1, height-1):
                if is_safe(i, j): continue
                if self.grid.get(i, j) is None:
                    # Usamos np_random de la clase padre
                    if self.np_random.uniform(0, 1) < 0.05: 
                        self.grid.set(i, j, Lava())

        # --- OBJETOS ---
        r0, r2, r5, r7 = rooms[0], rooms[2], rooms[5], rooms[7]

        # Room 0: Llave Amarilla
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(r0['x']+1, r0['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # Room 2: Puerta Amarilla / Llave Roja
        self.door_yellow = Door('yellow', is_open=False, is_locked=True)
        self.grid.set(r2['door_pos'][0], r2['door_pos'][1], self.door_yellow)
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(r2['x']+1, r2['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # Room 5: Puerta Roja / Llave Azul
        self.door_red = Door('red', is_open=False, is_locked=True)
        self.grid.set(r5['door_pos'][0], r5['door_pos'][1], self.door_red)
        self.key_blue = Key('blue')
        self.place_obj(self.key_blue, top=(r5['x']+1, r5['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # Room 7: Puerta Azul / Goal
        self.door_blue = Door('blue', is_open=False, is_locked=True)
        self.grid.set(r7['door_pos'][0], r7['door_pos'][1], self.door_blue)
        self.place_obj(Goal(), top=(r7['x']+1, r7['y']+1), size=(ROOM_S-2, ROOM_S-2))

        # --- SPAWN AGENT ---
        if self.curriculum_stage == 0:
            self.door_blue.is_locked = False; self.door_blue.is_open = True
            self.place_agent(top=(r7['x']+1, r7['y']+1), size=(ROOM_S-2, ROOM_S-2))
        elif self.curriculum_stage == 1:
            self.door_red.is_locked = False; self.door_red.is_open = True
            self.place_agent(top=(r5['x']+1, r5['y']+1), size=(ROOM_S-2, ROOM_S-2))
        elif self.curriculum_stage == 2:
            self.door_yellow.is_locked = False; self.door_yellow.is_open = True
            self.place_agent(top=(r2['x']+1, r2['y']+1), size=(ROOM_S-2, ROOM_S-2))
        elif self.curriculum_stage == 3:
            self.place_agent(top=(r0['x']+1, r0['y']+1), size=(ROOM_S-2, ROOM_S-2))
        else:
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

        # RECOMPENSAS
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            if not self.rewards_history['got_yellow']:
                reward += 0.5; self.rewards_history['got_yellow'] = True
        
        if not pre_yellow_open and self.door_yellow.is_open:
            if not self.rewards_history['opened_yellow']:
                reward += 1.0; self.rewards_history['opened_yellow'] = True
        
        if pre_carrying != self.key_red and self.carrying == self.key_red:
            if not self.rewards_history['got_red']:
                reward += 1.5; self.rewards_history['got_red'] = True

        if not pre_red_open and self.door_red.is_open:
            if not self.rewards_history['opened_red']:
                reward += 2.0; self.rewards_history['opened_red'] = True

        if pre_carrying != self.key_blue and self.carrying == self.key_blue:
            if not self.rewards_history['got_blue']:
                reward += 2.5; self.rewards_history['got_blue'] = True

        if not pre_blue_open and self.door_blue.is_open:
            if not self.rewards_history['opened_blue']:
                reward += 4.0; self.rewards_history['opened_blue'] = True
        
        return obs, reward, terminated, truncated, info

# Registrar el entorno en Gymnasium
try:
    register(id='MiniGrid-Ruins-v0', entry_point=RuinedTempleEnv)
except:
    pass

# ==========================================
# 2. EXTRACTOR DE CARACTERÍSTICAS (CNN)
# ==========================================
class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================
# 3. CALLBACK DE CURRICULUM
# ==========================================
class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.current_stage = 0
        self.max_stage = 4
        
        # Umbrales basados en tus recompensas
        self.promotion_thresholds = {
            0: 0.8,
            1: 6.0,
            2: 9.0,
            3: 11.0,
            4: float('inf')
        }

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            except:
                return True # Esperar a tener logs

            if len(x) > 0:
                mean_reward = np.mean(y[-40:]) # Media de los últimos 40 episodios
                
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps} | Stage: {self.current_stage} | Mean Reward: {mean_reward:.2f}")

                target = self.promotion_thresholds.get(self.current_stage, 9999)
                
                if mean_reward > target and self.current_stage < self.max_stage:
                    self.current_stage += 1
                    print(f"\n🎉 PROMOCIÓN! Subiendo a Stage {self.current_stage}")
                    
                    # Actualizar entorno vectorizado
                    self.training_env.env_method("set_curriculum_stage", self.current_stage)
                    
                    # Guardar checkpoint
                    self.model.save(f"{self.log_dir}/model_stage_{self.current_stage}")
                    
        return True

# ==========================================
# 4. MAIN & ENTRENAMIENTO
# ==========================================
def make_env(rank, seed=0, log_dir=None):
    def _init():
        # Inicializar entorno
        env = gym.make('MiniGrid-Ruins-v0', render_mode="rgb_array")
        # Wrapper para convertir la observación dict a imagen (Tensor)
        env = ImgObsWrapper(env) 
        # Monitor para logs
        if log_dir:
            env = VecMonitor(env) if False else env # Hack: Monitor se aplica en el vec_env o aquí
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

if __name__ == "__main__":
    # Configuración de Logs
    log_dir = "./ruins_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Crear entorno vectorizado (Multiproceso)
    num_cpu = 4
    # SubprocVecEnv es mejor para cargas pesadas, DummyVecEnv para debug
    env = SubprocVecEnv([make_env(i, log_dir=log_dir) for i in range(num_cpu)])

    # Configuración del Modelo
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # Inicializar RecurrentPPO
    model = RecurrentPPO(
        "CnnLstmPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        n_steps=512,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=0.0003,
        ent_coef=0.01,
        tensorboard_log="./ruins_tensorboard/"
    )

    # Callback
    curriculum_cb = CurriculumCallback(check_freq=2048, log_dir=log_dir)

    print("--- INICIANDO ENTRENAMIENTO ---")
    model.learn(total_timesteps=2_000_000, callback=curriculum_cb)

    model.save("ruined_temple_final")
    env.close()
    print("Entrenamiento finalizado.")