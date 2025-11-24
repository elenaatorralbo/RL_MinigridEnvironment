import gymnasium as gym
import numpy as np
import math
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper

# Cambiamos PPO por DQN
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import os
import time


# ==========================================
# 1. ENTORNO "PORTAL" CON RECOMPENSA DE DISTANCIA
# ==========================================
class PortalChamberEnv(MiniGridEnv):
    def __init__(self, difficulty=0, render_mode=None, size=8, max_steps=200, **kwargs):
        self.difficulty = difficulty

        # Posiciones importantes para calcular distancia
        self.key_pos = None
        self.door_pos = None
        self.goal_pos = None
        self.last_dist = 0  # Para calcular si se ha acercado o alejado

        # Flags de eventos
        self.rewarded_key = False
        self.rewarded_door = False

        mission_space = MissionSpace(mission_func=lambda: "Resuelve la prueba")
        if max_steps is None: max_steps = 4 * size ** 2

        super().__init__(mission_space=mission_space, grid_size=size, max_steps=max_steps, render_mode=render_mode,
                         **kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.rewarded_key = False
        self.rewarded_door = False

        # Calcular distancia inicial al primer objetivo (la llave)
        if self.difficulty >= 2 and self.key_pos:
            self.last_dist = self._get_dist(self.agent_pos, self.key_pos)
        else:
            self.last_dist = self._get_dist(self.agent_pos, self.goal_pos)

        return obs, info

    def _get_dist(self, pos_a, pos_b):
        """Distancia Manhattan (cuadricula)"""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.key_pos = None
        self.door_pos = None

        # --- CONFIGURACIÓN DE NIVELES ---

        # Nivel 0 y 1 (Sin llave)
        if self.difficulty < 2:
            self.place_agent()
            self.put_obj(Goal(), width - 2, height - 2)
            self.goal_pos = (width - 2, height - 2)
            if self.difficulty == 1:
                for _ in range(2): self.place_obj(Lava())

        # Nivel 2 y 3 (Con Llave y Puerta)
        elif self.difficulty >= 2:
            self.agent_start_pos = (1, 1)
            self.agent_dir = 0
            self.grid.set(1, 1, None)
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_dir

            # Muro central
            splitIdx = width // 2
            for i in range(0, height):
                self.grid.set(splitIdx, i, Wall())

            # Puerta
            doorIdx = self._rand_int(1, height - 1)
            self.door_pos = (splitIdx, doorIdx)
            self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))

            # Llave (Lado izquierdo)
            key_x = self._rand_int(1, splitIdx)
            key_y = self._rand_int(1, height - 1)
            # Evitar que caiga sobre el agente
            while (key_x, key_y) == self.agent_start_pos:
                key_x = self._rand_int(1, splitIdx)
                key_y = self._rand_int(1, height - 1)

            self.key_pos = (key_x, key_y)
            self.grid.set(key_x, key_y, Key('yellow'))

            # Meta (Lado derecho)
            self.put_obj(Goal(), width - 2, height - 2)
            self.goal_pos = (width - 2, height - 2)

            if self.difficulty >= 3:
                for _ in range(2): self.place_obj(Lava())

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # --- LOGICA DE SHAPING "FRIO / CALIENTE" ---
        # Solo aplicamos esto en dificultades con puzzle (Nivel 2+)
        if self.difficulty >= 2:

            # 1. DETERMINAR OBJETIVO ACTUAL
            current_target = self.goal_pos  # Por defecto

            has_key = self.carrying is not None and isinstance(self.carrying, Key)
            door_open = False
            if self.door_pos:
                door_obj = self.grid.get(*self.door_pos)
                door_open = door_obj is not None and door_obj.is_open

            if not has_key and not door_open:
                current_target = self.key_pos
            elif has_key and not door_open:
                current_target = self.door_pos
            else:
                current_target = self.goal_pos

            # 2. CALCULAR DISTANCIA Y RECOMPENSA
            # Si target es None (error de generación), ignoramos
            if current_target:
                dist = self._get_dist(self.agent_pos, current_target)

                # Reward = Diferencia de potencial (Me acerqué +0.1, Me alejé -0.1)
                # Esto evita que el agente haga trampas dando vueltas
                dist_reward = (self.last_dist - dist) * 0.1
                reward += dist_reward

                self.last_dist = dist

            # 3. BONUS POR EVENTOS (Grandes hitos)
            if has_key and not self.rewarded_key:
                reward += 1.0  # Bonus grande por coger llave
                self.rewarded_key = True

            if door_open and not self.rewarded_door:
                reward += 1.0  # Bonus grande por abrir puerta
                self.rewarded_door = True

        return obs, reward, terminated, truncated, info

    def set_difficulty(self, level):
        if self.difficulty != level: self.difficulty = level


# ==========================================
# 2. CALLBACK VISUAL
# ==========================================
class SmartCallback(BaseCallback):
    def __init__(self, check_freq=40000, verbose=0):
        super(SmartCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.current_difficulty = 0

    def _on_step(self) -> bool:
        # Curriculum Learning
        new_difficulty = 0
        if self.num_timesteps > 60000: new_difficulty = 1
        if self.num_timesteps > 150000: new_difficulty = 2

        if new_difficulty != self.current_difficulty:
            self.current_difficulty = new_difficulty
            print(f"\n[DQN] Subiendo dificultad a Nivel {new_difficulty}")
            env = self.training_env
            # Propagar dificultad a entornos vectorizados
            if hasattr(env, 'venv'): env = env.venv
            if hasattr(env, 'envs'):
                for e in env.envs:
                    curr = e
                    while hasattr(curr, 'env'):
                        if hasattr(curr, 'set_difficulty'):
                            curr.set_difficulty(new_difficulty)
                            break
                        curr = curr.env
                    if hasattr(curr, 'unwrapped') and hasattr(curr.unwrapped, 'set_difficulty'):
                        curr.unwrapped.set_difficulty(new_difficulty)

        if self.n_calls % self.check_freq == 0:
            self.show_preview()
        return True

    def show_preview(self):
        try:
            print(f"\n[VISUAL] DQN evaluando estrategia (Nivel {self.current_difficulty})...")
            eval_env = PortalChamberEnv(size=8, difficulty=self.current_difficulty, render_mode="human")
            eval_env = FlatObsWrapper(eval_env)

            obs, _ = eval_env.reset()
            for _ in range(2):
                done = False
                steps = 0
                while not done and steps < 100:
                    action, _ = self.model.predict(obs, deterministic=True)  # DQN suele ser determinista en test
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    eval_env.render()
                    # time.sleep(0.05)
                    steps += 1
                    if terminated or truncated:
                        done = True
                        obs, _ = eval_env.reset()
                        time.sleep(0.5)
            eval_env.close()
        except:
            pass


# ==========================================
# 3. EJECUCIÓN CON DQN
# ==========================================
def main():
    log_dir = "./portal_dqn_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print(">>> Iniciando Protocolo DQN con Reward Shaping (Distance)...")

    # DQN prefiere un solo entorno o pocos, pero para velocidad usamos 8
    vec_env = make_vec_env(lambda: FlatObsWrapper(PortalChamberEnv(size=8, difficulty=0, render_mode="rgb_array")),
                           n_envs=8)

    # CONFIGURACIÓN DE DQN
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        buffer_size=100000,  # Memoria de experiencias (Replay Buffer)
        learning_starts=1000,  # Pasos aleatorios al inicio para llenar memoria
        batch_size=32,
        exploration_fraction=0.1,  # Explorar el 10% del tiempo total
        exploration_final_eps=0.05,  # Nunca dejar de explorar un poquito (5%)
        device="cpu"
    )

    # Entrenamos
    total_steps = 400000
    model.learn(total_timesteps=total_steps, callback=SmartCallback(check_freq=50000))
    model.save("portal_agent_dqn")

    print("\n>>> MODO EXHIBICIÓN FINAL DQN <<<")
    final_env = PortalChamberEnv(size=8, difficulty=2, render_mode="human")
    final_env = FlatObsWrapper(final_env)

    obs, _ = final_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = final_env.step(action)
        final_env.render()
        time.sleep(0.1)

        if terminated or truncated:
            if reward > 2.0:  # 1 por llave + 1 por puerta + distancia
                print("¡PRUEBA SUPERADA CON ÉXITO!")
            obs, _ = final_env.reset()
            time.sleep(1)


if __name__ == "__main__":
    main()