import gymnasium as gym
import numpy as np
import os
import shutil
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


# ==========================================
# 1. CLASE PARA GUARDAR CHECKPOINTS (CORREGIDA)
# ==========================================
class CheckpointCallback(BaseCallback):
    """
    Callback personalizado para guardar el modelo cada X pasos.
    """

    def __init__(self, save_freq, save_path, verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Guardando modelo en {path}")
        return True


# ==========================================
# 2. CLASE DEL ENTORNO (Lógica Robusta)
# ==========================================
class PortalChamberEnv(MiniGridEnv):
    def __init__(self, difficulty=2, render_mode=None, size=8, max_steps=500, **kwargs):
        self.difficulty = difficulty
        self.mission_space = MissionSpace(mission_func=lambda: "Coge llave, abre puerta, cruza")
        self.splitIdx = size // 2
        self.last_dist = 0
        self.rewarded_key = False
        self.rewarded_open = False
        self.rewarded_cross = False
        if max_steps is None: max_steps = 4 * size ** 2
        super().__init__(mission_space=self.mission_space, grid_size=size, max_steps=max_steps, render_mode=render_mode,
                         **kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.rewarded_key = False
        self.rewarded_open = False
        self.rewarded_cross = False
        target = self._get_current_target()
        if target: self.last_dist = self._get_dist(self.agent_pos, target)
        return obs, info

    def _get_dist(self, pos_a, pos_b):
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def _get_current_target(self):
        has_key = self.carrying is not None and isinstance(self.carrying, Key)
        door_open = False
        if self.door_pos:
            d = self.grid.get(*self.door_pos)
            if d and d.is_open: door_open = True
        if not has_key:
            return self.key_pos
        elif not door_open:
            return self.door_pos
        else:
            return self.goal_pos

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.splitIdx = width // 2
        for i in range(0, height): self.grid.set(self.splitIdx, i, Wall())
        doorIdx = self._rand_int(1, height - 1)
        self.door_pos = (self.splitIdx, doorIdx)
        self.grid.set(self.splitIdx, doorIdx, Door('yellow', is_locked=True))

        left_top = (1, 1)
        left_size = (self.splitIdx - 1, height - 2)
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_pos = (width - 2, height - 2)
        key_obj = Key('yellow')
        self.place_obj(key_obj, top=left_top, size=left_size)
        self.key_pos = key_obj.cur_pos
        self.place_agent(top=left_top, size=left_size)
        if self.difficulty >= 3: self.place_obj(Lava(), top=left_top, size=left_size)

    def step(self, action):
        # Detectar estado previo para castigar acciones inútiles
        fwd_cell = self.grid.get(*self.front_pos)
        was_open = False
        if isinstance(fwd_cell, Door): was_open = fwd_cell.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # --- SISTEMA DE RECOMPENSAS ---
        target = self._get_current_target()
        if target:
            dist = self._get_dist(self.agent_pos, target)
            reward += (self.last_dist - dist) * 0.1
            self.last_dist = dist

        # ANTI-SPAM: Si pulsas toggle y no pasa nada -> CASTIGO
        if action == self.actions.toggle:
            is_open_now = False
            if isinstance(fwd_cell, Door): is_open_now = fwd_cell.is_open
            if was_open == is_open_now:
                reward -= 0.5  # ¡Deja de pulsar botones al aire!

        # BONUS
        has_key = self.carrying is not None and isinstance(self.carrying, Key)
        if has_key and not self.rewarded_key:
            reward += 3.0
            self.rewarded_key = True

        door_obj = self.grid.get(*self.door_pos)
        if door_obj and door_obj.is_open and not self.rewarded_open:
            reward += 5.0  # Gran premio por abrir
            self.rewarded_open = True

        if self.agent_pos[0] > self.splitIdx and not self.rewarded_cross:
            reward += 10.0  # ¡Premio GIGANTE por cruzar!
            self.rewarded_cross = True

        reward -= 0.02  # Coste por tiempo
        return obs, reward, terminated, truncated, info


# ==========================================
# 3. MAIN ENTRENAMIENTO
# ==========================================
def main():
    log_dir = "logs/portal_final"
    models_dir = "models/portal_final"

    # Limpieza inicial
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(">>> INICIANDO ENTRENAMIENTO FINAL (Anti-Spam + Jackpot)...")

    env = make_vec_env(lambda: FlatObsWrapper(PortalChamberEnv()), n_envs=8)

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=1e-4, buffer_size=100000, learning_starts=2000,
                batch_size=128, exploration_fraction=0.5, exploration_final_eps=0.05,
                target_update_interval=1000, device="auto")

    # Instanciamos nuestra clase CheckpointCallback correctamente
    callback = CheckpointCallback(save_freq=50000, save_path=models_dir)

    # Entrenar
    model.learn(total_timesteps=500000, callback=callback)

    # Guardar final
    model.save(f"{models_dir}/modelo_terminado")
    print(">>> ENTRENAMIENTO COMPLETADO. Ejecuta 'probar.py' ahora.")


if __name__ == "__main__":
    main()