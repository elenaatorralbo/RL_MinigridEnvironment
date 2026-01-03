import gymnasium as gym
import minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from gymnasium.envs.registration import register
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from tqdm.auto import tqdm


# =========================================================
# 1. WRAPPER 3 BOTONES (NECESARIO)
# =========================================================
class SimpleMovementWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, act):
        return act

    # =========================================================


# 2. ENTORNO: HUB QUANTUM (Teletransporte Agresivo)
# =========================================================
class HubQuantumEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19

        self.pos_key_red = (9, 9);
        self.pos_door_red = (9, 6)
        self.pos_key_blue = (9, 1);
        self.pos_door_blue = (12, 9)
        self.pos_key_yellow = (17, 9);
        self.pos_door_yellow = (6, 9)
        self.pos_goal = (1, 9)

        mission_space = MissionSpace(mission_func=lambda: "quantum leap navigation")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2500,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Muros
        h_left, h_right, h_top, h_bottom = 6, 12, 6, 12
        self.grid.vert_wall(h_left, 0, h_top);
        self.grid.horz_wall(0, h_top, h_left)
        self.grid.vert_wall(h_right, 0, h_top);
        self.grid.horz_wall(h_right, h_top, width - h_right)
        self.grid.vert_wall(h_left, h_bottom, height - h_bottom);
        self.grid.horz_wall(0, h_bottom, h_left)
        self.grid.vert_wall(h_right, h_bottom, height - h_bottom);
        self.grid.horz_wall(h_right, h_bottom, width - h_right)
        self.grid.wall_rect(h_left, h_top, 7, 7)

        # Puertas
        self.door_red = Door('red', is_locked=True);
        self.grid.set(*self.pos_door_red, self.door_red)
        self.door_blue = Door('blue', is_locked=True);
        self.grid.set(*self.pos_door_blue, self.door_blue)
        self.door_yellow = Door('yellow', is_locked=True);
        self.grid.set(*self.pos_door_yellow, self.door_yellow)

        # Objetos
        self.key_red = Key('red');
        self.grid.set(*self.pos_key_red, self.key_red)
        self.key_blue = Key('blue');
        self.grid.set(*self.pos_key_blue, self.key_blue)
        self.key_yellow = Key('yellow');
        self.grid.set(*self.pos_key_yellow, self.key_yellow)
        self.place_obj(Goal(), top=self.pos_goal, size=(1, 1))

        # --- EMBUDO ESTRECHO (Lava por todas partes) ---
        # Sala Norte
        for y in range(1, h_top):
            self.grid.set(7, y, Lava());
            self.grid.set(8, y, Lava())
            self.grid.set(10, y, Lava());
            self.grid.set(11, y, Lava())
        # Sala Este
        for x in range(h_right + 1, width - 1):
            self.grid.set(x, 7, Lava());
            self.grid.set(x, 8, Lava())
            self.grid.set(x, 10, Lava());
            self.grid.set(x, 11, Lava())
        # Sala Oeste
        for x in range(1, h_left):
            self.grid.set(x, 6, Lava());
            self.grid.set(x, 7, Lava());
            self.grid.set(x, 8, Lava())
            self.grid.set(x, 10, Lava());
            self.grid.set(x, 11, Lava());
            self.grid.set(x, 12, Lava())
        # Sur y Esquinas
        for x in range(h_left + 1, h_right):
            for y in range(h_bottom + 1, height - 1): self.grid.set(x, y, Lava())
        self.grid.set(h_left + 1, h_top + 1, Lava());
        self.grid.set(h_right - 1, h_top + 1, Lava())
        self.grid.set(h_left + 1, h_bottom - 1, Lava());
        self.grid.set(h_right - 1, h_bottom - 1, Lava())

        # Spawn (8,9)
        self.agent_pos = (8, 9)
        self.agent_dir = 0
        self.grid.set(8, 9, None)

    def reset(self, *, seed=None, options=None):
        self.has_red = False;
        self.opened_red = False
        self.has_blue = False;
        self.opened_blue = False
        self.has_yellow = False;
        self.opened_yellow = False
        obs, info = super().reset(seed=seed, options=options)

        self.target_pos = self._get_target_pos()
        self.prev_dist = self._get_dist_to(self.target_pos)
        return obs, info

    def _get_target_pos(self):
        if not self.has_red: return np.array(self.pos_key_red)
        if not self.opened_red: return np.array(self.pos_door_red)
        if not self.has_blue: return np.array(self.pos_key_blue)
        if not self.opened_blue: return np.array(self.pos_door_blue)
        if not self.has_yellow: return np.array(self.pos_key_yellow)
        if not self.opened_yellow: return np.array(self.pos_door_yellow)
        return np.array(self.pos_goal)

    def _get_dist_to(self, target):
        return np.sum(np.abs(np.array(self.agent_pos) - target))

    def dist_between(self, pos_a, pos_b):
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        state_changed = False

        # --- 1. IMÁN DE LLAVES + TELETRANSPORTE A PUERTA ---
        if not self.has_red and self.dist_between(self.agent_pos, self.pos_key_red) <= 1:
            self.has_red = True;
            self.grid.set(*self.pos_key_red, None)
            # TP a Puerta Roja
            self.agent_pos = (9, 8);
            self.agent_dir = 3;
            reward += 20.0;
            state_changed = True

        elif not self.has_blue and self.dist_between(self.agent_pos, self.pos_key_blue) <= 1:
            self.has_blue = True;
            self.grid.set(*self.pos_key_blue, None)
            # TP a Puerta Azul (Ojo, ahora tiene que bajar, lo encaramos al Sur)
            # Corrección: Después de coger azul, tiene que volver al centro.
            # Lo ponemos en (9, 3) mirando al Sur (1) para que baje
            self.agent_pos = (9, 3);
            self.agent_dir = 1;
            reward += 20.0;
            state_changed = True

        elif not self.has_yellow and self.dist_between(self.agent_pos, self.pos_key_yellow) <= 1:
            self.has_yellow = True;
            self.grid.set(*self.pos_key_yellow, None)
            # TP a Puerta Amarilla (Oeste)
            # Lo ponemos en (15, 9) mirando al Oeste (2) para que vuelva al centro
            self.agent_pos = (15, 9);
            self.agent_dir = 2;
            reward += 20.0;
            state_changed = True

        # --- 2. AUTO-OPEN PUERTAS + TELETRANSPORTE A OBJETIVO (SALTO CUÁNTICO) ---
        front_cell = self.grid.get(*self.front_pos)
        if action == self.actions.forward and front_cell and front_cell.type == 'door':

            # PUERTA ROJA -> SALTO A LA LLAVE AZUL
            if front_cell.color == 'red' and self.has_red:
                self.door_red.is_open = True;
                self.opened_red = True;
                self.has_red = False
                # LA CLAVE: Lo ponemos en (9, 3). La llave azul está en (9, 1). ¡Está a 2 pasos!
                self.agent_pos = (9, 3);
                self.agent_dir = 3;
                reward += 20.0;
                state_changed = True
                print(">>> SALTO CUÁNTICO A LA LLAVE AZUL")

            # PUERTA AZUL -> SALTO A LA LLAVE AMARILLA
            elif front_cell.color == 'blue' and self.has_blue:
                self.door_blue.is_open = True;
                self.opened_blue = True;
                self.has_blue = False
                # La llave amarilla está en (17, 9). Lo ponemos en (15, 9).
                self.agent_pos = (15, 9);
                self.agent_dir = 0;
                reward += 20.0;
                state_changed = True
                print(">>> SALTO CUÁNTICO A LA LLAVE AMARILLA")

            # PUERTA AMARILLA -> SALTO A LA META
            elif front_cell.color == 'yellow' and self.has_yellow:
                self.door_yellow.is_open = True;
                self.opened_yellow = True;
                self.has_yellow = False
                # La meta está en (1, 9). Lo ponemos en (3, 9).
                self.agent_pos = (3, 9);
                self.agent_dir = 2;
                reward += 20.0;
                state_changed = True
                print(">>> SALTO CUÁNTICO A LA META")

        # --- REWARDS ---
        if terminated and reward > 0: reward += 100.0

        reward -= 0.01

        self.target_pos = self._get_target_pos()
        dist_now = self._get_dist_to(self.target_pos)

        if state_changed:
            self.prev_dist = dist_now
        else:
            reward += (self.prev_dist - dist_now) * 2.0
            self.prev_dist = dist_now

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-HubQuantum-v17', entry_point='__main__:HubQuantumEnv')
except:
    pass

# =========================================================
# 3. ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-HubQuantum-v17"
    TOTAL_TIMESTEPS = 600_000

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=lambda e: FlatObsWrapper(SimpleMovementWrapper(e)))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.02,
        gamma=0.99,
        device="cpu"
    )


    class ProgressBar(BaseCallback):
        def _on_training_start(self): self.pbar = tqdm(total=self.locals['total_timesteps'])

        def _on_step(self): self.pbar.update(self.training_env.num_envs); return True

        def _on_training_end(self): self.pbar.close()


    print(f"🚀 Iniciando (Modo: SALTO CUÁNTICO)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressBar(TOTAL_TIMESTEPS))
    model.save("ppo_hub_quantum")

    # Visualizar
    print("--- Testeando ---")
    env = gym.make(env_id, render_mode="human")
    env = SimpleMovementWrapper(env)
    env = FlatObsWrapper(env)

    model = PPO.load("ppo_hub_quantum", device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            if reward > 0: print(">>> ¡VICTORIA! <<<")
            obs, _ = env.reset()