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
import numpy as np
from tqdm.auto import tqdm


# =========================================================
# 1. BARRA DE PROGRESO
# =========================================================
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando", unit="steps")

    def _on_step(self) -> bool:
        if self.pbar: self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar: self.pbar.close()


# =========================================================
# 2. ENTORNO: ANTI-CONGELAMIENTO (SPAWN FIJO)
# =========================================================
class HubMagicKeysEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19

        # Posiciones
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
            max_steps=2500,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Geometría Cruz
        h_left, h_right = 6, 12
        h_top, h_bottom = 6, 12

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

        # Lava
        for y in range(1, h_top): self.grid.set(7, y, Lava()); self.grid.set(11, y, Lava())
        for x in range(h_right + 1, width - 1):
            self.grid.set(x, 7, Lava());
            self.grid.set(x, 11, Lava())
            if x % 2 == 0: self.grid.set(x, 8, Lava()); self.grid.set(x, 10, Lava())
        self.grid.set(2, 8, Lava());
        self.grid.set(2, 10, Lava());
        self.grid.set(3, 9, Lava())
        for x in range(1, h_left): self.grid.set(x, 6, Lava()); self.grid.set(x, 12, Lava())
        self.grid.set(h_left + 1, h_top + 1, Lava());
        self.grid.set(h_right - 1, h_top + 1, Lava())
        self.grid.set(h_left + 1, h_bottom - 1, Lava());
        self.grid.set(h_right - 1, h_bottom - 1, Lava())
        for x in range(h_left + 1, h_right):
            for y in range(h_bottom + 1, height - 1):
                if (x + y) % 2 == 0: self.grid.set(x, y, Lava())

        # --- AGENTE (CORRECCIÓN CRÍTICA) ---
        # 1. Posición Fija: Justo a la izquierda de la llave (8,9)
        # 2. Dirección Fija: Mirando al Este (0), es decir, mirando a la llave.
        # Esto elimina la confusión inicial.
        self.agent_pos = (8, 9)
        self.agent_dir = 0
        self.grid.set(8, 9, None)  # Asegurar que la casilla está vacía para el agente

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
        agent_pos = np.array(self.agent_pos)
        return np.sum(np.abs(agent_pos - target))

    def step(self, action):
        # 1. Logear acción para debug (Solo print si se atasca mucho)
        # if action == self.actions.pickup: print("INTENTO COGER")

        dist_before = self._get_dist_to(self.target_pos)

        # Estado Puertas
        door_red_was_closed = not self.door_red.is_open
        door_blue_was_closed = not self.door_blue.is_open
        door_yellow_was_closed = not self.door_yellow.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        state_changed = False

        # --- LLAVES MÁGICAS ---
        if door_red_was_closed and self.door_red.is_open:
            self.carrying = None;
            self.opened_red = True;
            reward += 10.0;
            state_changed = True
        elif door_blue_was_closed and self.door_blue.is_open:
            self.carrying = None;
            self.opened_blue = True;
            reward += 10.0;
            state_changed = True
        elif door_yellow_was_closed and self.door_yellow.is_open:
            self.carrying = None;
            self.opened_yellow = True;
            reward += 10.0;
            state_changed = True

        # --- RECOLECCIÓN ---
        if not self.has_red and self.carrying and self.carrying.type == 'key' and self.carrying.color == 'red':
            self.has_red = True;
            reward += 10.0;
            state_changed = True
        elif self.opened_red and not self.has_blue and self.carrying and self.carrying.type == 'key' and self.carrying.color == 'blue':
            self.has_blue = True;
            reward += 10.0;
            state_changed = True
        elif self.opened_blue and not self.has_yellow and self.carrying and self.carrying.type == 'key' and self.carrying.color == 'yellow':
            self.has_yellow = True;
            reward += 10.0;
            state_changed = True

        if terminated and reward > 0: reward += 100.0

        # --- REWARD SHAPING ---
        if state_changed:
            self.target_pos = self._get_target_pos()
            self.prev_dist = self._get_dist_to(self.target_pos)
            reward += 5.0
        else:
            dist_now = self._get_dist_to(self.target_pos)
            # Recompensa más agresiva por moverse
            reward += (self.prev_dist - dist_now) * 1.0
            self.prev_dist = dist_now

        reward -= 0.005

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-HubUnfrozen-v10', entry_point='__main__:HubMagicKeysEnv')
except:
    pass

# =========================================================
# 3. ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-HubUnfrozen-v10"

    # 800k pasos es un buen equilibrio
    TOTAL_TIMESTEPS = 800_000

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    # SUBIMOS ENTROPÍA A 0.05
    # Esto es vital: Obliga al agente a probar teclas aleatorias (como Coger)
    # aunque crea que no sirve de nada.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.05,  # <--- MAXIMA CURIOSIDAD
        gamma=0.99,
        device="cpu"
    )

    bar_callback = ProgressBarCallback(TOTAL_TIMESTEPS)

    print(f"🚀 Iniciando (Modo: Descongelación)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=bar_callback)

    model.save("ppo_hub_unfrozen")
    print("✅ Guardado.")

    # Visualizar
    print("--- Testeando ---")
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)
    model = PPO.load("ppo_hub_unfrozen", device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            if reward > 0: print(">>> ¡VICTORIA! <<<")
            obs, _ = env.reset()