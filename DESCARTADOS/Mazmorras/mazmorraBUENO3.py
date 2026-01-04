import gymnasium as gym
import numpy as np
import os
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces
from gymnasium.envs.registration import register

# Wrappers
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
# IMPORTANTE: Librerías de estabilidad
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize 

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sb3_contrib import RecurrentPPO

# ==========================================
# 1. ENTORNO (3 FASES)
# ==========================================
class CurriculumTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None, max_steps=1000):
        self.grid_w = 19
        self.grid_h = 19
        self.current_level = 1 
        
        # Historial de recompensas (Objetos)
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False, 
            'got_red': False, 'opened_red': False, 
            'got_blue': False, 'opened_blue': False
        }
        
        # Historial de Habitaciones Visitadas (NUEVO)
        self.visited_rooms = set()
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the temple")
        
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=max_steps, 
            render_mode=render_mode
        )
        
        self.action_space = spaces.Discrete(6)

    def set_level(self, level):
        if self.current_level != level:
            print(f"\n🆙 NIVEL ACTUALIZADO: {self.current_level} -> {level}")
            self.current_level = level

    def reset(self, *, seed=None, options=None):
        self.rewards_history = {k: False for k in self.rewards_history}
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Reseteamos las habitaciones visitadas al empezar la partida
        self.visited_rooms = set()
        # Añadimos la habitación inicial como "ya vista"
        start_room = (self.agent_pos[0] // 6, self.agent_pos[1] // 6)
        self.visited_rooms.add(start_room)
        
        return obs, info

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(6, 0); self.grid.vert_wall(12, 0)
        self.grid.horz_wall(0, 6); self.grid.horz_wall(0, 12)

        # FASE 1: Zig-Zag sin puertas (Huecos abiertos estratégicos)
        if self.current_level == 1:
            # Huecos que fuerzan el zig-zag (Derecha -> Izquierda -> Derecha)
            self.grid.set(6, 3, None)   # Paso 1: Ir a la derecha (arriba)
            self.grid.set(6, 15, None)  # Paso 2: Volver a la izquierda (abajo) NO, espera
            # Para un zig-zag real en 3 columnas:
            # Col 1 -> Col 2 (por arriba)
            # Col 2 -> Col 3 (por abajo)
            self.grid.set(6, 3, None)   # Hueco Pared 1 (Arriba)
            self.grid.set(12, 15, None) # Hueco Pared 2 (Abajo)
            
            # Paredes extra para forzar el camino si quieres
            self.grid.set(12, 3, Wall()) # Cerrar arriba para obligar a bajar
            
            self.place_agent(top=(1, 1), size=(5, 17)) # Spawn a la izquierda

        # FASE 2: PUZZLE (Igual que antes)
        if self.current_level >= 2:
            self.door_yellow = Door('yellow', is_locked=True); self.grid.set(15, 6, self.door_yellow) 
            self.door_red = Door('red', is_locked=True); self.grid.set(3, 12, self.door_red)
            self.door_blue = Door('blue', is_locked=True); self.grid.set(12, 15, self.door_blue)

            self.key_yellow = Key('yellow'); self.place_obj(self.key_yellow, top=(7, 1), size=(5, 5))
            self.key_red = Key('red'); self.place_obj(self.key_red, top=(7, 7), size=(5, 5))
            self.key_blue = Key('blue'); self.place_obj(self.key_blue, top=(7, 13), size=(5, 5))
            
            self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
            self.place_agent(top=(1, 1), size=(5, 5))

        # FASE 3: LAVA
        if self.current_level >= 3:
            lava_density = 0.25
            for col_room in range(3):
                for row_room in range(3):
                    if col_room == 2 and row_room == 2: continue 
                    base_x = col_room * 6; base_y = row_room * 6
                    for x in range(base_x + 2, base_x + 5):
                        for y in range(base_y + 2, base_y + 5):
                            if self.grid.get(x, y) is None:
                                if self.np_random.uniform(0, 1) < lava_density:
                                    self.grid.set(x, y, Lava())

        self.place_obj(Goal(), top=(13, 13), size=(5, 5)) # Meta abajo derecha
        self.mission = "traverse the temple"

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # MOVIMIENTO
        target_pos = None
        if action == 0: self.agent_dir = 2; target_pos = self.agent_pos + np.array([-1, 0])
        elif action == 1: self.agent_dir = 0; target_pos = self.agent_pos + np.array([1, 0])
        elif action == 2: self.agent_dir = 3; target_pos = self.agent_pos + np.array([0, -1])
        elif action == 3: self.agent_dir = 1; target_pos = self.agent_pos + np.array([0, 1])

        if target_pos is not None:
            cell = self.grid.get(*target_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = target_pos
            if cell is not None and cell.type == 'lava':
                terminated = True; reward = -10

        elif action == 4: # PICKUP
            fwd_pos = self.front_pos; fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell; self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    if self.current_level >= 2:
                        if self.carrying.color == 'yellow': reward += 5.0; self.rewards_history['got_yellow'] = True
                        elif self.carrying.color == 'red': reward += 5.0; self.rewards_history['got_red'] = True
                        elif self.carrying.color == 'blue': reward += 5.0; self.rewards_history['got_blue'] = True

        elif action == 5: # TOGGLE
            fwd_pos = self.front_pos; fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.type == 'door' and fwd_cell.is_locked:
                if self.carrying and self.carrying.color == fwd_cell.color:
                    fwd_cell.is_locked = False; fwd_cell.is_open = True; self.carrying = None
                    if self.current_level >= 2:
                        if fwd_cell.color == 'yellow': reward += 5.0; self.rewards_history['opened_yellow'] = True
                        elif fwd_cell.color == 'red': reward += 5.0; self.rewards_history['opened_red'] = True
                        elif fwd_cell.color == 'blue': reward += 5.0; self.rewards_history['opened_blue'] = True

        # --- 🚀 RECOMPENSA POR EXPLORACIÓN DE HABITACIONES (Room Checkpoints) ---
        # El mapa es 19x19. Las paredes están en 6 y 12.
        # Dividimos el mapa en coordenadas de "habitaciones" (0,0), (1,0), (2,2), etc.
        current_room_x = self.agent_pos[0] // 6
        current_room_y = self.agent_pos[1] // 6
        current_room = (current_room_x, current_room_y)

        if current_room not in self.visited_rooms:
            reward += 2.0 # ¡Premio por descubrir zona nueva!
            self.visited_rooms.add(current_room)
        # ------------------------------------------------------------------------

        if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            terminated = True; reward += 20.0 # Meta final

        if self.step_count >= self.max_steps: truncated = True

        return self.gen_obs(), reward, terminated, truncated, {}

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
        
        self.thresholds = {
            1: 45.0,   
            2: 90.0,   
            3: 999.0   
        }

    def _on_step(self) -> bool:
        return True

    def update_level(self, mean_reward):
        if self.current_level >= 3: return
        required_score = self.thresholds[self.current_level]
        
        if mean_reward >= required_score:
            print(f"\n🚀 ¡SUBIDA DE NIVEL! Puntuación media: {mean_reward:.2f} >= {required_score}")
            print(f"Pasando de Fase {self.current_level} a {self.current_level + 1}")
            self.current_level += 1
            self.my_train_env.env_method("set_level", self.current_level)
            self.eval_env.env_method("set_level", self.current_level)
            self.my_train_env.reset()

# --- AQUÍ ESTABA EL ERROR: FALTABA ESTA CLASE ---
class SmartEvalCallback(EvalCallback):
    def __init__(self, curriculum_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_cb = curriculum_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        # Verificamos si acaba de terminar una evaluación
        if self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.curriculum_cb.update_level(self.last_mean_reward)
        return result
# -----------------------------------------------

# ==========================================
# 3. SETUP ESTABILIZADO
# ==========================================
if __name__ == "__main__":
    NUM_ENVS = 16 
    TOTAL_TIMESTEPS = 5_000_000
    LOG_DIR = "./CurriculumSymbolic3/Logs"
    MODEL_DIR = "./CurriculumSymbolic3/Models"
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    def make_symbolic_env():
        env = gym.make('MiniGrid-Curriculum-Symbolic-v0')
        env = FullyObsWrapper(env)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env

    # 1. Crear entornos
    train_env = make_vec_env(make_symbolic_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(make_symbolic_env, n_envs=1, vec_env_cls=SubprocVecEnv)

    # 2. Normalización (Vital para estabilidad)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0)
    eval_env.training = False; eval_env.norm_reward = False

    curriculum_manager = CurriculumManagerCallback(eval_env=eval_env, train_env=train_env)

    smart_eval_callback = SmartEvalCallback(
        curriculum_cb=curriculum_manager,
        eval_env=eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=2000, 
        n_eval_episodes=10,
        deterministic=False
    )
    
    # 3. Modelo Configurado (Lento y Seguro)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4, 
        ent_coef=0.005,     
        n_steps=128,
        batch_size=256,
        gamma=0.99,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1
        )
    )

    print("\n⚡ INICIANDO ENTRENAMIENTO ESTABILIZADO (3 FASES) ⚡")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=smart_eval_callback, progress_bar=True)
        model.save(os.path.join(MODEL_DIR, "Final_Symbolic"))
        train_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    except KeyboardInterrupt:
        model.save(os.path.join(MODEL_DIR, "Interrupted_Symbolic"))
        train_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    finally:
        train_env.close()
        eval_env.close()