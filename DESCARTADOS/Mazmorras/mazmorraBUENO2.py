import gymnasium as gym
import numpy as np
import os
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register

# --- WRAPPERS NUEVOS (CLAVE PARA VELOCIDAD) ---
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sb3_contrib import RecurrentPPO

# ==========================================
# 1. ENTORNO (Lógica idéntica, solo cambia el Wrapper al final)
# ==========================================
class CurriculumTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19
        self.current_level = 1 
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the temple")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=1000, 
            render_mode=render_mode
        )

    def set_level(self, level):
        if self.current_level != level:
            print(f"--- Environment switching to Level {level} ---")
            self.current_level = level

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # PAREDES
        self.grid.vert_wall(6, 0); self.grid.vert_wall(12, 0)
        self.grid.horz_wall(0, 6); self.grid.horz_wall(0, 12)

        # HUECOS
        self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
        self.grid.set(12, 9, None); self.grid.set(6, 9, None) 
        self.grid.set(6, 15, None) 

        # PUERTAS
        self.door_yellow = Door('yellow', is_locked=True); self.grid.set(15, 6, self.door_yellow) 
        self.door_red = Door('red', is_locked=True); self.grid.set(3, 12, self.door_red)
        self.door_blue = Door('blue', is_locked=True); self.grid.set(12, 15, self.door_blue)

        # LLAVES
        self.key_yellow = Key('yellow'); self.place_obj(self.key_yellow, top=(7, 1), size=(5, 5))
        self.key_red = Key('red'); self.place_obj(self.key_red, top=(7, 7), size=(5, 5))
        self.key_blue = Key('blue'); self.place_obj(self.key_blue, top=(7, 13), size=(5, 5))

        # META
        self.place_obj(Goal(), top=(13, 13), size=(5, 5))
        self.goal_pos = np.array([15, 15]) 

        # LAVA
        lava_density = 0.3
        for col_room in range(3):
            for row_room in range(3):
                if col_room == 2 and row_room == 2: continue 
                base_x = col_room * 6
                base_y = row_room * 6
                for x in range(base_x + 2, base_x + 5):
                    for y in range(base_y + 2, base_y + 5):
                        if self.grid.get(x, y) is None:
                            if self.np_random.uniform(0, 1) < lava_density:
                                self.grid.set(x, y, Lava())

        # SPAWN
        if self.current_level == 1: self.place_agent(top=(13, 13), size=(5, 5))
        elif self.current_level == 2: self.place_agent(top=(1, 13), size=(5, 5))
        elif self.current_level == 3: self.place_agent(top=(13, 7), size=(5, 5))
        else: self.place_agent(top=(1, 1), size=(5, 5))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.rewards_history = {'got_yellow': False, 'opened_yellow': False, 'got_red': False, 'opened_red': False, 'got_blue': False, 'opened_blue': False}
        
        target_pos = self._get_target_pos()
        # CORRECCIÓN AQUI: np.array()
        self.prev_dist = np.abs(np.array(self.agent_pos) - np.array(target_pos)).sum()
        
        return obs, info

    def _get_target_pos(self):
        """
        Lógica Mágica: Devuelve la coordenada (x,y) de lo que el agente DEBERÍA perseguir.
        INCLUYE PROTECCIÓN CONTRA 'NONE' (Objetos en inventario).
        """
        
        def safe_pos(obj):
            # Si el objeto está en el tablero, devuelve su posición.
            # Si el objeto está en el bolsillo (pos es None), devuelve la posición del agente 
            # (así la distancia es 0 y no explota).
            if obj.cur_pos is not None:
                return obj.cur_pos
            return self.agent_pos

        # Nivel 1: Ir directo a la meta
        if self.current_level == 1:
            return self.goal_pos

        # Nivel 2: Llave Azul -> Puerta Azul -> Meta
        if self.current_level == 2:
            if not self.rewards_history['got_blue']:
                return safe_pos(self.key_blue)
            elif not self.door_blue.is_open:
                return safe_pos(self.door_blue)
            else:
                return self.goal_pos

        # Nivel 3: Llave Roja -> Puerta Roja -> Llave Azul -> P. Azul -> Meta
        if self.current_level == 3:
            if not self.rewards_history['got_red']:
                return safe_pos(self.key_red)
            elif not self.door_red.is_open:
                return safe_pos(self.door_red)
            elif not self.rewards_history['got_blue']:
                return safe_pos(self.key_blue)
            elif not self.door_blue.is_open:
                return safe_pos(self.door_blue)
            else:
                return self.goal_pos

        # Nivel 4: Amarilla -> Roja -> Azul -> Meta
        if self.current_level == 4:
            if not self.rewards_history['got_yellow']:
                return safe_pos(self.key_yellow)
            elif not self.door_yellow.is_open:
                return safe_pos(self.door_yellow)
            elif not self.rewards_history['got_red']:
                return safe_pos(self.key_red)
            elif not self.door_red.is_open:
                return safe_pos(self.door_red)
            elif not self.rewards_history['got_blue']:
                return safe_pos(self.key_blue)
            elif not self.door_blue.is_open:
                return safe_pos(self.door_blue)
            else:
                return self.goal_pos
        
        return self.goal_pos

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open; pre_red_open = self.door_red.is_open; pre_blue_open = self.door_blue.is_open
        
        old_target_pos = self._get_target_pos()

        obs, reward, terminated, truncated, info = super().step(action)

        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow and not self.rewards_history['got_yellow']:
            reward += 10.0; self.rewards_history['got_yellow'] = True
        if pre_carrying != self.key_red and self.carrying == self.key_red and not self.rewards_history['got_red']:
            reward += 10.0; self.rewards_history['got_red'] = True
        if pre_carrying != self.key_blue and self.carrying == self.key_blue and not self.rewards_history['got_blue']:
            reward += 10.0; self.rewards_history['got_blue'] = True

        if not pre_yellow_open and self.door_yellow.is_open and not self.rewards_history['opened_yellow']:
            reward += 10.0; self.rewards_history['opened_yellow'] = True
        if not pre_red_open and self.door_red.is_open and not self.rewards_history['opened_red']:
            reward += 10.0; self.rewards_history['opened_red'] = True
        if not pre_blue_open and self.door_blue.is_open and not self.rewards_history['opened_blue']:
            reward += 10.0; self.rewards_history['opened_blue'] = True
        new_target_pos = self._get_target_pos()
        
        # CORRECCIÓN AQUI: np.array() en las comparaciones y restas
        if not np.array_equal(np.array(old_target_pos), np.array(new_target_pos)):
            self.prev_dist = np.abs(np.array(self.agent_pos) - np.array(new_target_pos)).sum()
        
        curr_dist = np.abs(np.array(self.agent_pos) - np.array(new_target_pos)).sum()
        reward += (self.prev_dist - curr_dist) * 0.1
        self.prev_dist = curr_dist

        if terminated and self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            reward += 50.0

        return obs, reward, terminated, truncated, info

# Registrar el entorno
if 'MiniGrid-Curriculum-Symbolic-v0' in gym.envs.registry:
    del gym.envs.registry['MiniGrid-Curriculum-Symbolic-v0']
register(id='MiniGrid-Curriculum-Symbolic-v0', entry_point='__main__:CurriculumTempleEnv')

# ==========================================
# 2. CURRICULUM MANAGER
# ==========================================
class CurriculumManagerCallback(BaseCallback):
    def __init__(self, eval_env, train_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.my_train_env = train_env
        self.current_level = 1
        
        # Umbrales ajustados para simbólico (aprenderá más rápido y puntuará mejor)
        self.thresholds = {
            1: 50.0,   # Para pasar de nivel 1 a 2
            2: 65.0,  # Para pasar de nivel 2 a 3
            3: 85.0,  # Para pasar de nivel 3 a 4
            4: 105.0   # Nivel máximo (Masterizado)
        }

    def _on_step(self) -> bool:
        return True

    def update_level(self, mean_reward):
        if self.current_level >= 4: return
        required_score = self.thresholds[self.current_level]
        if mean_reward >= required_score:
            print(f"\n🚀 CURRICULUM UPGRADE! Reward {mean_reward:.2f} >= {required_score}")
            print(f"Promoting from Level {self.current_level} to {self.current_level + 1}")
            self.current_level += 1
            self.my_train_env.env_method("set_level", self.current_level)
            self.eval_env.env_method("set_level", self.current_level)
            self.my_train_env.reset(); self.eval_env.reset()

# ==========================================
# 3. SETUP Y ENTRENAMIENTO
# ==========================================
if __name__ == "__main__":
    NUM_ENVS = 16  # ¡Podemos usar más entornos paralelos porque es simbólico y gasta menos CPU!
    TOTAL_TIMESTEPS = 5_000_000 # Necesitará menos pasos que el visual
    LOG_DIR = "./CurriculumSymbolic/Logs"
    MODEL_DIR = "./CurriculumSymbolic/Models"
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    # --- WRAPPER MAGIC ---
    # Aquí está el truco: FullyObs + FlatObs
    def make_symbolic_env():
        env = gym.make('MiniGrid-Curriculum-Symbolic-v0')
        env = FullyObsWrapper(env) # 1. El agente ve TODO el mapa (19x19)
        env = FlatObsWrapper(env)  # 2. Convierte el mapa en un vector de números (0,1,2,5...)
        return env

    # Creamos los entornos vectorizados usando la función de arriba
    train_env = make_vec_env(make_symbolic_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    # Para evaluación usamos el mismo wrapper setup
    eval_env = make_vec_env(make_symbolic_env, n_envs=1, vec_env_cls=SubprocVecEnv)

    curriculum_manager = CurriculumManagerCallback(eval_env=eval_env, train_env=train_env)

    # Callback de Evaluación
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
        eval_freq=1000, # Evaluar frecuentemente
        n_eval_episodes=10,
        deterministic=False
    )
    
    # --- CAMBIO IMPORTANTE: MlpLstmPolicy ---
    # Al usar FlatObsWrapper, la entrada es un vector plano de números.
    # No necesitamos CNN (features_extractor). Usamos MLP directo.
    
    model = RecurrentPPO(
        "MlpLstmPolicy",  # <--- MLP en vez de CNN (Mucho más rápido)
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=5e-4, # Un poco más alto para aprender rápido
        ent_coef=0.03,
        n_steps=128,       # Pasos más cortos por update
        batch_size=256,
        gamma=0.99,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1
        )
    )

    print("\n⚡ INICIANDO ENTRENAMIENTO SIMBÓLICO (MODO TURBO) ⚡")
    print("Observación: Mapa completo (FullyObs) convertido a números (FlatObs)")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=smart_eval_callback, progress_bar=True)
        model.save(os.path.join(MODEL_DIR, "Final_Symbolic"))
    except KeyboardInterrupt:
        model.save(os.path.join(MODEL_DIR, "Interrupted_Symbolic"))
    finally:
        train_env.close()
        eval_env.close()