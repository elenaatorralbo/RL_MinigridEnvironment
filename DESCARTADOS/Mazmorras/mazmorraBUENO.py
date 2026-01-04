import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sb3_contrib import RecurrentPPO

class CurriculumTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19
        
        # --- NIVEL DE CURRICULUM ---
        # 1: Final Room (Solo Meta)
        # 2: Blue Key Room (Llave Azul -> Puerta Azul -> Meta)
        # 3: Red Key Room (Llave Roja -> P. Roja -> Llave Azul -> P. Azul -> Meta)
        # 4: Full Run (Amarilla -> Roja -> Azul -> Meta)
        self.current_level = 1 
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the temple")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2000, 
            render_mode=render_mode
        )

    def set_level(self, level):
        """Método para actualizar el nivel desde el Callback"""
        # Solo imprimimos si cambia el nivel para no saturar la consola
        if self.current_level != level:
            print(f"--- Environment switching to Level {level} ---")
            self.current_level = level

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- PAREDES ESTRUCTURALES ---
        self.grid.vert_wall(6, 0); self.grid.vert_wall(12, 0)
        self.grid.horz_wall(0, 6); self.grid.horz_wall(0, 12)

        # --- HUECOS ---
        self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
        self.grid.set(12, 9, None); self.grid.set(6, 9, None) 
        self.grid.set(6, 15, None) 

        # --- OBJETOS ---
        # Puertas
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(15, 6, self.door_yellow) 
        self.door_red = Door('red', is_locked=True)
        self.grid.set(3, 12, self.door_red)
        self.door_blue = Door('blue', is_locked=True)
        self.grid.set(12, 15, self.door_blue)

        # Llaves
        self.key_yellow = Key('yellow'); self.place_obj(self.key_yellow, top=(7, 1), size=(5, 5))
        self.key_red = Key('red'); self.place_obj(self.key_red, top=(7, 7), size=(5, 5))
        self.key_blue = Key('blue'); self.place_obj(self.key_blue, top=(7, 13), size=(5, 5))

        # Meta
        # Colocamos la meta. place_obj busca un hueco libre en la zona definida.
        self.place_obj(Goal(), top=(13, 13), size=(5, 5))
        # Nota: La posición exacta puede variar dentro de la sala 13,13, pero la lógica de distancia usa el centro aprox.
        self.goal_pos = np.array([15, 15]) 

        # --- LAVA (Peligros) - LÓGICA CORREGIDA ---
        # Solo en el 3x3 central de cada habitación para dejar pasillos seguros pegados a las paredes.
        lava_density = 0.3 # Aumentamos un poco la densidad ya que el área es menor
        
        # Iteramos sobre las 9 habitaciones (3 columnas x 3 filas)
        for col_room in range(3):
            for row_room in range(3):
                # Coordenada base (esquina superior izquierda de las paredes de la habitación)
                base_x = col_room * 6
                base_y = row_room * 6
                
                # El interior de la habitación va de base+1 a base+5.
                # El CENTRO 3x3 va de base+2 a base+4 (indices 2, 3, 4 relativos a la pared).
                for x in range(base_x + 2, base_x + 5):
                    for y in range(base_y + 2, base_y + 5):

                        if col_room == 2 and row_room == 2:
                            continue # Saltamos esta habitación, no ponemos lava aquí
                        
                        # Verificamos que no haya nada (ni puertas, ni llaves, ni meta ya colocada)
                        if self.grid.get(x, y) is None:
                            if self.np_random.uniform(0, 1) < lava_density:
                                self.grid.set(x, y, Lava())

        # --- SPAWN LOGIC SEGÚN CURRICULUM ---
        if self.current_level == 1:
            # FASE 1: Sala de Victoria (Abajo-Derecha)
            self.place_agent(top=(13, 13), size=(5, 5))
            
        elif self.current_level == 2:
            # FASE 2: Sala Llave Azul (Abajo-Izquierda)
            self.place_agent(top=(1, 13), size=(5, 5))
            
        elif self.current_level == 3:
            # FASE 3: Sala Llave Roja (Medio-Derecha)
            self.place_agent(top=(13, 7), size=(5, 5))
            
        else: # Level 4
            # FASE 4: Full Hardcore (Arriba-Izquierda)
            self.place_agent(top=(1, 1), size=(5, 5))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False,
            'got_red': False, 'opened_red': False,
            'got_blue': False, 'opened_blue': False
        }
        self.prev_dist = np.abs(self.agent_pos - self.goal_pos).sum()
        return obs, info

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open
        pre_blue_open = self.door_blue.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # --- RECOMPENSAS SOLICITADAS ---
        
        # 1. SHAPING POR DISTANCIA (+- 0.1)
        curr_dist = np.abs(self.agent_pos - self.goal_pos).sum()
        dist_diff = self.prev_dist - curr_dist 
        
        if dist_diff > 0:
            reward += 0.1
        elif dist_diff < 0:
            reward -= 0.1
        
        self.prev_dist = curr_dist

        # 2. VICTORIA (+8)
        if terminated and self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            reward += 8.0 

        # 3. LLAVES (+3)
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            if not self.rewards_history['got_yellow']: 
                reward += 3.0; self.rewards_history['got_yellow'] = True

        if pre_carrying != self.key_red and self.carrying == self.key_red:
            if not self.rewards_history['got_red']: 
                reward += 3.0; self.rewards_history['got_red'] = True

        if pre_carrying != self.key_blue and self.carrying == self.key_blue:
            if not self.rewards_history['got_blue']: 
                reward += 3.0; self.rewards_history['got_blue'] = True

        # 4. PUERTAS (+3)
        if not pre_yellow_open and self.door_yellow.is_open:
            if not self.rewards_history['opened_yellow']: 
                reward += 3.0; self.rewards_history['opened_yellow'] = True

        if not pre_red_open and self.door_red.is_open:
            if not self.rewards_history['opened_red']: 
                reward += 3.0; self.rewards_history['opened_red'] = True

        if not pre_blue_open and self.door_blue.is_open:
            if not self.rewards_history['opened_blue']: 
                reward += 3.0; self.rewards_history['opened_blue'] = True

        return obs, reward, terminated, truncated, info

register(id='MiniGrid-Curriculum-v0', entry_point='__main__:CurriculumTempleEnv')

# ==========================================
# 2. RED NEURONAL (VISION + MEMORIA)
# ==========================================
class CurriculumManagerCallback(BaseCallback):
    # AÑADIMOS train_env AL CONSTRUCTOR
    def __init__(self, eval_env, train_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.my_train_env = train_env # <--- LO GUARDAMOS CON OTRO NOMBRE
        self.current_level = 2
        
        # DEFINICIÓN DE UMBRALES (THRESHOLDS)
        self.thresholds = {
            1: 8.95,   # Para pasar de nivel 1 a 2
            2: 13.0,  # Para pasar de nivel 2 a 3
            3: 18.0,  # Para pasar de nivel 3 a 4
            4: 21.0   # Nivel máximo (Masterizado)
        }

    def _on_step(self) -> bool:
        return True

    def update_level(self, mean_reward):
        if self.current_level >= 4:
            return # Ya estamos en el nivel máximo

        required_score = self.thresholds[self.current_level]
        
        if mean_reward >= required_score:
            print(f"\n🚀 CURRICULUM UPGRADE! Reward {mean_reward:.2f} >= {required_score}")
            print(f"Promoting from Level {self.current_level} to {self.current_level + 1}")
            
            self.current_level += 1
            
            # USAMOS NUESTRA VARIABLE PROPIA my_train_env
            self.my_train_env.env_method("set_level", self.current_level)
            self.eval_env.env_method("set_level", self.current_level)
            
            # Reseteamos los entornos para aplicar cambios inmediatos
            self.my_train_env.reset()
            self.eval_env.reset()

# --- RED NEURONAL (Igual que antes) ---
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

if __name__ == "__main__":
    NUM_ENVS = 8  
    TOTAL_TIMESTEPS = 6_000_000 
    LOG_DIR = "./Curriculum/Logs"
    MODEL_DIR = "./Curriculum/Models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Entornos
    train_env = make_vec_env('MiniGrid-Curriculum-v0', n_envs=NUM_ENVS, wrapper_class=ImgObsWrapper, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env('MiniGrid-Curriculum-v0', n_envs=1, wrapper_class=ImgObsWrapper, vec_env_cls=SubprocVecEnv)

    # 2. Callback Manager (Pasando train_env correctamente)
    curriculum_manager = CurriculumManagerCallback(eval_env=eval_env, train_env=train_env)

    # 3. Callback Eval Wrapper
    class SmartEvalCallback(EvalCallback):
        def __init__(self, curriculum_cb, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.curriculum_cb = curriculum_cb
            
        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.n_calls % self.eval_freq == 0: 
                self.curriculum_cb.update_level(self.last_mean_reward)
            return result

    smart_eval_callback = SmartEvalCallback(
        curriculum_cb=curriculum_manager,
        eval_env=eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=1000, 
        n_eval_episodes=20,
        deterministic=False, 
        render=False
    )
    
    # 4. Modelo
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        lstm_hidden_size=256,
        n_lstm_layers=1
    )

    path_to_model = "./Curriculum/Models/Curriculum_Interrupted.zip" # <--- Asegúrate de que esta ruta sea correcta (el archivo .zip)

    model = RecurrentPPO.load(
        path_to_model,
        env=train_env, # Es crucial pasar el entorno aquí para re-vincularlo
        verbose=1,
        tensorboard_log=LOG_DIR, # Para seguir graficando
        # learning_rate=3e-4,    # Opcional: SB3 suele cargar el LR guardado, pero puedes forzarlo si quieres
        # ent_coef=0.03          # Opcional: Igual que arriba
    )

    print("\n🎓 INICIANDO CURRICULUM LEARNING (Final) 🎓")
    START_LEVEL = 2

    # Actualizamos el entorno de entrenamiento
    train_env.env_method("set_level", START_LEVEL)
    train_env.reset()

    # Actualizamos el entorno de evaluación
    eval_env.env_method("set_level", START_LEVEL)
    eval_env.reset()

    print(f"\n🔄 RESUMIENDO ENTRENAMIENTO DESDE EL NIVEL {START_LEVEL}")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=smart_eval_callback, progress_bar=True)
        model.save(os.path.join(MODEL_DIR, "Curriculum_Final"))
    except KeyboardInterrupt:
        model.save(os.path.join(MODEL_DIR, "Curriculum_Interrupted"))
    finally:
        train_env.close()
        eval_env.close()