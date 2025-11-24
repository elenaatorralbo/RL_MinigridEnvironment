import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper

# Librerías de RL
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import os
import time


# ==========================================
# 1. ENTORNO PORTAL (Con correcciones)
# ==========================================
class PortalChamberEnv(MiniGridEnv):
    def __init__(self, difficulty=0, render_mode=None, size=8, max_steps=100, **kwargs):
        self.difficulty = difficulty
        self.agent_start_pos = (1, 1)
        self.agent_dir = 0

        # Misión fija con lambda para evitar errores de versión
        mission_space = MissionSpace(mission_func=lambda: "Resuelve la prueba")

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # NIVEL 0: Básico
        if self.difficulty == 0:
            self.place_agent()
            self.place_obj(Goal())

        # NIVEL 1: Lava
        elif self.difficulty == 1:
            self.place_agent()
            self.place_obj(Goal())
            for _ in range(2):
                self.place_obj(Lava())

        # NIVEL 2: Llave y Puerta
        elif self.difficulty == 2:
            self.agent_start_pos = (1, 1)
            self.agent_dir = 0
            self.grid.set(1, 1, None)
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_dir

            splitIdx = width // 2
            for i in range(0, height):
                self.grid.set(splitIdx, i, Wall())

            # Puerta y llave
            doorIdx = self._rand_int(1, height - 1)
            self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))

            key_pos = (self._rand_int(1, splitIdx), self._rand_int(1, height - 1))
            while key_pos == self.agent_start_pos:
                key_pos = (self._rand_int(1, splitIdx), self._rand_int(1, height - 1))
            self.grid.set(key_pos[0], key_pos[1], Key('yellow'))

            self.put_obj(Goal(), width - 2, height - 2)

        # NIVEL 3: Caos
        elif self.difficulty >= 3:
            self.place_agent()
            self.place_obj(Goal())
            for _ in range(3):
                self.place_obj(Lava())
            for _ in range(width):
                self.place_obj(Wall())

    def set_difficulty(self, level):
        if self.difficulty != level:
            self.difficulty = level


# ==========================================
# 2. CALLBACK HÍBRIDO (CURRICULUM + VISUAL)
# ==========================================
class SmartCallback(BaseCallback):
    """
    Controla la dificultad Y muestra una ventana visual cada X pasos.
    """

    def __init__(self, check_freq=50000, verbose=0):
        super(SmartCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.current_difficulty = 0

    def _on_step(self) -> bool:
        # 1. Lógica de Curriculum (Subir nivel)
        new_difficulty = 0
        if self.num_timesteps > 80000: new_difficulty = 1
        if self.num_timesteps > 150000: new_difficulty = 2

        if new_difficulty != self.current_difficulty:
            self.current_difficulty = new_difficulty
            print(f"\n[GLaDOS] ¡Dificultad aumentada al Nivel {new_difficulty}!")
            # Actualizar entornos de entrenamiento
            env = self.training_env
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

        # 2. Lógica Visual (Mostrar ventana cada X pasos)
        if self.n_calls % self.check_freq == 0:
            print(f"\n[VISUAL] Pausando entrenamiento para mostrar progreso (Step {self.num_timesteps})...")
            self.show_preview()

        return True

    def show_preview(self):
        """Crea un entorno temporal solo para ver cómo juega"""
        try:
            # Creamos un entorno visual con la dificultad actual
            eval_env = PortalChamberEnv(size=8, difficulty=self.current_difficulty, render_mode="human")
            eval_env = FlatObsWrapper(eval_env)

            obs, _ = eval_env.reset()
            for _ in range(3):  # Mostrar 3 episodios
                done = False
                while not done:
                    # Usamos el modelo actual para predecir
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    eval_env.render()  # Dibuja la ventana
                    # time.sleep(0.05)  # Pequeña pausa para que el ojo humano lo vea

                    if terminated or truncated:
                        done = True
                        obs, _ = eval_env.reset()
                        time.sleep(0.5)

            eval_env.close()
            print("[VISUAL] Ventana cerrada. Reanudando entrenamiento rápido...\n")
        except Exception as e:
            print(f"No se pudo renderizar: {e}")


# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    log_dir = "./portal_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print(">>> Iniciando GLaDOS Training System (CPU Mode)...")
    print(">>> Se abrirá una ventana cada 20,000 pasos para que veas el progreso.")

    # Entorno vectorizado (8 a la vez para velocidad)
    def make_env():
        env = PortalChamberEnv(size=8, difficulty=0, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        return env

    vec_env = make_vec_env(make_env, n_envs=8)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
        ent_coef=0.01,
        device="cpu"  # Forzamos CPU
    )

    # Entrenar (Configurado para mostrar visualmente cada 20k pasos)
    total_steps = 300000
    callback = SmartCallback(check_freq=20000)  # <--- CAMBIA ESTO si quieres ver más/menos frecuente

    model.learn(total_timesteps=total_steps, callback=callback)
    model.save("portal_agent_final")

    # -------------------------------------------------
    # VISUALIZACIÓN FINAL INFINITA
    # -------------------------------------------------
    print("\n>>> ENTRENAMIENTO FINALIZADO <<<")
    print(">>> Iniciando Modo Exhibición (Ctrl+C para salir)...")

    final_env = PortalChamberEnv(size=8, difficulty=2, render_mode="human")
    final_env = FlatObsWrapper(final_env)

    obs, _ = final_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = final_env.step(action)
        final_env.render()
        time.sleep(0.1)

        if terminated or truncated:
            if reward > 0: print("¡Pastel conseguido!")
            obs, _ = final_env.reset()
            time.sleep(1)


if __name__ == "__main__":
    main()