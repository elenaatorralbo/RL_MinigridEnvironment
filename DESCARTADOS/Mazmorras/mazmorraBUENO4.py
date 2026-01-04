import gymnasium as gym
import numpy as np
import os
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sb3_contrib import RecurrentPPO

# ==========================================
# 1. ENTORNO ESPIRAL (4 Habitaciones)
# ==========================================
class SpiralTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 11
        self.grid_h = 11
        self.current_level = 1 
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the spiral temple")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=1000, # Más pasos porque el recorrido es más largo
            render_mode=render_mode
        )

    def set_level(self, level):
        if self.current_level != level:
            print(f"--- Environment switching to Level {level} ---")
            self.current_level = level

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- PAREDES (Cruz central) ---
        self.grid.vert_wall(5, 0)
        self.grid.horz_wall(0, 5)

        # --- PUERTAS Y HUECOS ---
        # 1. Puerta Amarilla (Arriba: Izq -> Der)
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(5, 2, self.door_yellow) 

        # 2. Puerta Roja (Derecha: Arriba -> Abajo)
        self.door_red = Door('red', is_locked=True)
        self.grid.set(8, 5, self.door_red)
        
        # 3. HUECO (Abajo: Der -> Izq) - Para entrar a la meta
        self.grid.set(5, 8, None) 

        # --- LLAVES ---
        # Llave Amarilla (Inicio)
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(1, 1), size=(4, 4))

        # Llave Roja (Segunda habitación)
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(6, 1), size=(4, 4))

        # --- META (AHORA EN ABAJO-IZQUIERDA) ---
        self.place_obj(Goal(), top=(1, 6), size=(4, 4))
        # Buscamos la pos exacta
        for x in range(width):
            for y in range(height):
                obj = self.grid.get(x, y)
                if obj and obj.type == 'goal':
                    self.goal_pos = np.array([x, y])

        # --- LAVA ---
        # Un poco de lava en la habitación de paso (Abajo-Derecha)
        self.grid.set(7, 7, Lava())

        # --- SPAWN POR NIVEL ---
        if self.current_level == 1:
            # FASE 1: Empieza en la habitación de la Meta (Abajo-Izquierda)
            self.place_agent(top=(1, 6), size=(4, 4))
        
        elif self.current_level == 2:
            # FASE 2: Empieza en Arriba-Derecha (Llave Roja -> Puerta Roja -> Paso -> Meta)
            self.place_agent(top=(6, 1), size=(4, 4))
            # La amarilla ya está abierta para no molestar
            self.door_yellow.is_open = True 
            
        else: 
            # FASE 3: Recorrido Completo (Arriba-Izquierda)
            self.place_agent(top=(1, 1), size=(4, 4))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.rewards_history = {'got_yellow': False, 'opened_yellow': False, 'got_red': False, 'opened_red': False}
        
        target_pos = self._get_target_pos()
        self.prev_dist = np.abs(np.array(self.agent_pos) - np.array(target_pos)).sum()
        return obs, info

    def _get_target_pos(self):
        def safe_pos(obj):
            if obj.cur_pos is not None: return obj.cur_pos
            return self.agent_pos

        # Lógica de guiado actualizada para la espiral
        if self.current_level == 1:
            return self.goal_pos

        if self.current_level == 2:
            if not self.rewards_history['got_red']: return safe_pos(self.key_red)
            elif not self.door_red.is_open: return safe_pos(self.door_red)
            else: return self.goal_pos

        if self.current_level >= 3:
            if not self.rewards_history['got_yellow']: return safe_pos(self.key_yellow)
            elif not self.door_yellow.is_open: return safe_pos(self.door_yellow)
            elif not self.rewards_history['got_red']: return safe_pos(self.key_red)
            elif not self.door_red.is_open: return safe_pos(self.door_red)
            else: return self.goal_pos
        
        return self.goal_pos

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow = self.door_yellow.is_open; pre_red = self.door_red.is_open
        old_target = self._get_target_pos()

        obs, reward, terminated, truncated, info = super().step(action)

        # RECOMPENSAS
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow and not self.rewards_history['got_yellow']:
            reward += 10.0; self.rewards_history['got_yellow'] = True
        if pre_carrying != self.key_red and self.carrying == self.key_red and not self.rewards_history['got_red']:
            reward += 10.0; self.rewards_history['got_red'] = True
        if not pre_yellow and self.door_yellow.is_open and not self.rewards_history['opened_yellow']:
            reward += 10.0; self.rewards_history['opened_yellow'] = True
        if not pre_red and self.door_red.is_open and not self.rewards_history['opened_red']:
            reward += 10.0; self.rewards_history['opened_red'] = True

        # Shaping de distancia
        new_target = self._get_target_pos()
        if not np.array_equal(old_target, new_target):
             self.prev_dist = np.abs(self.agent_pos - new_target).sum()
        curr_dist = np.abs(self.agent_pos - new_target).sum()
        reward += (self.prev_dist - curr_dist) * 0.1
        self.prev_dist = curr_dist

        if terminated and self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            reward += 50.0 # Meta vale más porque es más difícil llegar

        return obs, reward, terminated, truncated, info

if 'MiniGrid-Spiral-v0' in gym.envs.registry: del gym.envs.registry['MiniGrid-Spiral-v0']
register(id='MiniGrid-Spiral-v0', entry_point='__main__:SpiralTempleEnv')

# ==========================================
# 2. CURRICULUM & TRAINING
# ==========================================
class SpiralCurriculumCallback(BaseCallback):
    def __init__(self, eval_env, train_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env; self.train_env = train_env
        self.current_level = 1
        self.thresholds = {1: 40.0, 2: 60.0, 3: 150.0} # Ajustado a las nuevas recompensas

    def _on_step(self) -> bool: return True
    def update_level(self, mean_reward):
        if self.current_level >= 3: return
        if mean_reward >= self.thresholds[self.current_level]:
            print(f"\n🌀 UPGRADING TO SPIRAL LEVEL {self.current_level + 1}!")
            self.current_level += 1
            self.train_env.env_method("set_level", self.current_level)
            self.eval_env.env_method("set_level", self.current_level)
            self.train_env.reset()

if __name__ == "__main__":
    NUM_ENVS = 16 
    TOTAL_TIMESTEPS = 2_000_000 # Un poco más porque el camino es más largo
    LOG_DIR = "./SpiralTemple/Logs"; MODEL_DIR = "./SpiralTemple/Models"
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    def make_env():
        env = gym.make('MiniGrid-Spiral-v0')
        return FlatObsWrapper(FullyObsWrapper(env))

    train_env = make_vec_env(make_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)

    curriculum = SpiralCurriculumCallback(eval_env, train_env)

    eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_freq=500, n_eval_episodes=10)
    # Monkey patch para conectar el curriculum al eval
    original_on_step = eval_cb._on_step
    def patched_on_step():
        cont = original_on_step()
        if eval_cb.n_calls % eval_cb.eval_freq == 0:
            curriculum.update_level(eval_cb.last_mean_reward)
        return cont
    eval_cb._on_step = patched_on_step

    model = RecurrentPPO("MlpLstmPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=3e-4, n_steps=256, batch_size=256, policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=1))

    print("\n🌀 STARTING SPIRAL TRAINING 🌀")
    model.learn(TOTAL_TIMESTEPS, callback=eval_cb, progress_bar=True)
    model.save(f"{MODEL_DIR}/Final_Spiral_Agent")