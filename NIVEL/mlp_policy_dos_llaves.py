import gymnasium as gym
import minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os


# =========================================================
# 1. ENTORNO CUSTOM: 2 LLAVES, 2 PUERTAS, 3 ZONAS
# =========================================================
class TwoKeysEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 16  # Mapa grande
        self.grid_h = 16

        mission_space = MissionSpace(
            mission_func=lambda: "find yellow key, open door, swap for red key, open door, goal")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=1000,  # Damos tiempo suficiente para la logística de soltar/coger
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Dividimos el mapa en 3 habitaciones verticales
        # Zona 1: Inicio | Zona 2: Intermedia | Zona 3: Meta

        x_wall_1 = width // 3
        x_wall_2 = (width // 3) * 2
        centerY = height // 2

        # Muro 1
        self.grid.vert_wall(x_wall_1, 0, height)
        # Muro 2
        self.grid.vert_wall(x_wall_2, 0, height)

        # --- FASE 1: AMARILLO ---
        # Puerta Amarilla en Muro 1
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(x_wall_1, centerY, self.door_yellow)

        # Llave Amarilla en Zona 1
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(1, 1), size=(x_wall_1 - 1, height - 2))

        # --- FASE 2: ROJO ---
        # Puerta Roja en Muro 2
        self.door_red = Door('red', is_locked=True)
        self.grid.set(x_wall_2, centerY, self.door_red)

        # Llave Roja en Zona 2 (Entre los dos muros)
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(x_wall_1 + 1, 1), size=((x_wall_2 - x_wall_1) - 1, height - 2))

        # --- FASE 3: META ---
        # Meta en Zona 3
        self.place_obj(Goal(), top=(x_wall_2 + 1, 1), size=((width - x_wall_2) - 1, height - 2))

        # Agente en Zona 1
        self.place_agent(top=(1, 1), size=(x_wall_1 - 1, height - 2))

    def reset(self, *, seed=None, options=None):
        # Reiniciamos los "Checkpoints" de recompensas
        self.got_yellow = False
        self.opened_yellow = False
        self.got_red = False
        self.opened_red = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # --- SISTEMA DE RECOMPENSAS ESCALONADO ---

        # 1. Coger Llave Amarilla (+5)
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            if not self.got_yellow:
                reward += 5.0
                self.got_yellow = True
                # print(">> Yellow Key")

        # 2. Abrir Puerta Amarilla (+10)
        if not pre_yellow_open and self.door_yellow.is_open:
            if not self.opened_yellow:
                reward += 10.0
                self.opened_yellow = True
                # print(">> Yellow Door")

        # 3. Coger Llave Roja (+15) - AQUÍ ESTÁ EL TRUCO
        # Para coger esta, el agente habrá tenido que soltar la amarilla antes.
        # Le damos más puntos para motivarle a hacer ese intercambio.
        if pre_carrying != self.key_red and self.carrying == self.key_red:
            if not self.got_red:
                reward += 15.0
                self.got_red = True
                # print(">> Red Key (Swapped!)")

        # 4. Abrir Puerta Roja (+20)
        if not pre_red_open and self.door_red.is_open:
            if not self.opened_red:
                reward += 20.0
                self.opened_red = True
                # print(">> Red Door")

        # 5. Meta (+50)
        if terminated and reward > 0:
            reward += 50.0

        return obs, reward, terminated, truncated, info


# Registrar
try:
    register(id='MiniGrid-TwoKeys-v0', entry_point='__main__:TwoKeysEnv')
except:
    pass

# =========================================================
# 2. ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-TwoKeys-v0"

    # IMPORTANTE: Usamos FlatObsWrapper (Vectores, no imágenes)
    # Esto le da al agente info directa: "Inventory: Yellow Key", "Inventory: Empty".
    # Es crucial para que aprenda a soltar la llave.

    print(f"--- Entrenando 'Double Trouble' (16x16, 2 Llaves) ---")

    # 8 entornos paralelos para velocidad
    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        device="auto"
    )

    # Necesitamos más pasos porque la lógica de "soltar llave" es difícil de descubrir
    total_timesteps = 2_000_000
    model.learn(total_timesteps=total_timesteps)

    save_path = "PPO_TwoKeys_Flat"
    model.save(save_path)
    print("--- Entrenamiento finalizado ---")

    # =========================================================
    # 3. VISUALIZACIÓN
    # =========================================================
    print("--- Visualizando la estrategia del agente ---")

    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(save_path)

    obs, _ = env.reset()
    print("Observa: Debería soltar la llave amarilla después de usarla.")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            if reward > 0:
                print(">>> ¡Misión Cumplida! <<<")
            obs, _ = env.reset()