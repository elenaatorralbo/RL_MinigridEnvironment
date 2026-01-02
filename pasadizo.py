import gymnasium as gym
import minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import numpy as np  # Necesario para calcular distancias
from tqdm.auto import tqdm


# =========================================================
# 1. UTILIDAD: BARRA DE PROGRESO
# =========================================================
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando Agente", unit="steps")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


# =========================================================
# 2. ENTORNO: HUB LAVA (CON DISTANCE SHAPING)
# =========================================================
class HubGlobalLavaEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19

        # Guardamos las posiciones fijas de los objetivos para calcular distancias
        # (x, y)
        self.pos_key_red = (9, 9)
        self.pos_door_red = (9, 6)

        self.pos_key_blue = (9, 1)
        self.pos_door_blue = (12, 9)

        self.pos_key_yellow = (17, 9)
        self.pos_door_yellow = (6, 9)

        self.pos_goal = (1, 9)

        mission_space = MissionSpace(mission_func=lambda: "avoid lava use keys to reach goal")

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

        # --- GEOMETRÍA CRUZ ---
        h_left, h_right = 6, 12
        h_top, h_bottom = 6, 12

        self.grid.vert_wall(h_left, 0, h_top)
        self.grid.horz_wall(0, h_top, h_left)
        self.grid.vert_wall(h_right, 0, h_top)
        self.grid.horz_wall(h_right, h_top, width - h_right)
        self.grid.vert_wall(h_left, h_bottom, height - h_bottom)
        self.grid.horz_wall(0, h_bottom, h_left)
        self.grid.vert_wall(h_right, h_bottom, height - h_bottom)
        self.grid.horz_wall(h_right, h_bottom, width - h_right)
        self.grid.wall_rect(h_left, h_top, 7, 7)

        # --- PUERTAS ---
        self.door_red = Door('red', is_locked=True)
        self.grid.set(*self.pos_door_red, self.door_red)

        self.door_blue = Door('blue', is_locked=True)
        self.grid.set(*self.pos_door_blue, self.door_blue)

        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(*self.pos_door_yellow, self.door_yellow)

        # --- OBJETOS ---
        self.key_red = Key('red')
        self.grid.set(*self.pos_key_red, self.key_red)

        self.key_blue = Key('blue')
        self.grid.set(*self.pos_key_blue, self.key_blue)

        self.key_yellow = Key('yellow')
        self.grid.set(*self.pos_key_yellow, self.key_yellow)

        self.place_obj(Goal(), top=self.pos_goal, size=(1, 1))

        # --- LAVA ---
        # (Tu configuración de lava se mantiene igual, omitida por brevedad visual,
        # asegúrate de usar tu código de lava aquí)
        # ... [Pegar tu código de lava aquí] ...
        # Para que funcione el ejemplo rápido pondré solo un poco:
        for y in range(1, h_top):
            self.grid.set(7, y, Lava())
            self.grid.set(11, y, Lava())

        # --- AGENTE ---
        self.place_agent(top=(8, 8), size=(3, 3))

    def reset(self, *, seed=None, options=None):
        # Reiniciar estados
        self.has_red = False;
        self.opened_red = False
        self.has_blue = False;
        self.opened_blue = False
        self.has_yellow = False;
        self.opened_yellow = False

        obs, info = super().reset(seed=seed, options=options)

        # Inicializar distancia previa al primer objetivo
        self.prev_dist = self._get_dist_to_target()
        return obs, info

    def _get_dist_to_target(self):
        """Calcula distancia Manhattan al objetivo actual."""
        agent_pos = np.array(self.agent_pos)

        # Determinar objetivo actual
        if not self.has_red:
            target = np.array(self.pos_key_red)
        elif not self.opened_red:
            target = np.array(self.pos_door_red)
        elif not self.has_blue:
            target = np.array(self.pos_key_blue)
        elif not self.opened_blue:
            target = np.array(self.pos_door_blue)
        elif not self.has_yellow:
            target = np.array(self.pos_key_yellow)
        elif not self.opened_yellow:
            target = np.array(self.pos_door_yellow)
        else:
            target = np.array(self.pos_goal)

        return np.sum(np.abs(agent_pos - target))

    def step(self, action):
        pre_carrying = self.carrying
        pre_d_red = self.door_red.is_open
        pre_d_blue = self.door_blue.is_open
        pre_d_yellow = self.door_yellow.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # --- LOGICA DE ESTADOS ---
        # Detectar cambios de estado para las recompensas grandes
        if not self.has_red and self.carrying and self.carrying.type == 'key' and self.carrying.color == 'red':
            self.has_red = True
            reward += 10.0  # Subimos recompensa
        elif self.has_red and not self.opened_red and self.door_red.is_open:
            self.opened_red = True
            reward += 10.0

        elif self.opened_red and not self.has_blue and self.carrying and self.carrying.type == 'key' and self.carrying.color == 'blue':
            self.has_blue = True
            reward += 10.0
        elif self.has_blue and not self.opened_blue and self.door_blue.is_open:
            self.opened_blue = True
            reward += 10.0

        elif self.opened_blue and not self.has_yellow and self.carrying and self.carrying.type == 'key' and self.carrying.color == 'yellow':
            self.has_yellow = True
            reward += 10.0
        elif self.has_yellow and not self.opened_yellow and self.door_yellow.is_open:
            self.opened_yellow = True
            reward += 10.0

        if terminated and reward > 0:
            reward += 100.0

        # --- SHAPING REWARD (EL IMÁN) ---
        # Calculamos nueva distancia al objetivo actual
        curr_dist = self._get_dist_to_target()

        # Si la distancia bajó (nos acercamos), recompensa pequeña positiva
        if curr_dist < self.prev_dist:
            reward += 0.1
        # Si la distancia subió (nos alejamos), penalización pequeña
        elif curr_dist > self.prev_dist:
            reward -= 0.1

        # Actualizamos la distancia previa para el siguiente step
        self.prev_dist = curr_dist

        # Pequeña penalización por paso para incentivar velocidad
        reward -= 0.01

        return obs, reward, terminated, truncated, info


# Registrar entorno
try:
    register(id='MiniGrid-HubLavaPro-v7', entry_point='__main__:HubGlobalLavaEnv')
except:
    pass

# =========================================================
# 3. ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-HubLavaPro-v7"

    # IMPORTANTE: Aumentar timesteps. 1 es insuficiente.
    # 500,000 es un buen comienzo para tareas de navegación complejas.
    TOTAL_TIMESTEPS = 500000

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,  # Un poco menos de entropía si usamos shaping
        gamma=0.99,
        device="cpu"
    )

    bar_callback = ProgressBarCallback(TOTAL_TIMESTEPS)

    print(f"🚀 Iniciando entrenamiento PRO con Shaping de Distancia...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=bar_callback)

    model.save("ppo_hub_solved")
    print("✅ Guardado.")

    # Visualizar
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    # Cargar modelo recién entrenado
    model = PPO.load("ppo_hub_solved", device="cpu")

    obs, _ = env.reset()
    print("--- Testeando Agente ---")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()