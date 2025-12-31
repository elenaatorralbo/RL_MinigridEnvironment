import gymnasium as gym
import minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Ball
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np


# =========================================================
# 1. ENTORNO: ENDLESS RUNNER (BUCLE INFINITO)
# =========================================================
class EndlessSurvivalEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 20  # Mapa corto para que las rondas sean rápidas
        self.grid_h = 9

        # El objetivo cambia dinámicamente, así que ponemos uno genérico
        mission_space = MissionSpace(mission_func=lambda: "survive as many rounds as possible")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=5000,  # Mucho tiempo, queremos ver hasta dónde llega
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        # Esta función se llama al principio de todo (Reset real)
        # Inicializamos el contador de rondas
        if not hasattr(self, 'round_count'):
            self.round_count = 1

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Generamos el nivel basado en la dificultad actual
        self._generate_level_obstacles()

        # Agente al inicio
        self.place_agent(top=(1, 1), size=(2, height - 2))

        # Meta visual (Puerta verde al final)
        self.grid.set(width - 2, height // 2, Goal())

    def _generate_level_obstacles(self):
        """Genera obstáculos procedimentales según la ronda actual"""
        width, height = self.grid.width, self.grid.height

        # Limpiamos zona de juego (por si venimos de una ronda anterior)
        # Dejamos las paredes (col 0 y col width-1) intactas
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                self.grid.set(i, j, None)

        # Volvemos a poner la meta
        self.grid.set(width - 2, height // 2, Goal())

        # --- DIFICULTAD PROGRESIVA ---
        # Ronda 1: Fácil
        # Ronda 2: Lava
        # Ronda 3: Monstruo
        # Ronda 10: EL INFIERNO

        difficulty = self.round_count

        # 1. LAVA (Aumenta con las rondas)
        num_lava = int(difficulty * 2)  # Ronda 1=2 lavas, Ronda 5=10 lavas
        for _ in range(num_lava):
            self.place_obj(Lava(), top=(3, 1), size=(width - 5, height - 2), max_tries=100)

        # 2. MONSTRUOS (Bolas perseguidoras)
        # Aparece 1 monstruo cada 2 rondas. (Ronda 1=0, Ronda 2=1, Ronda 4=2...)
        num_monsters = difficulty // 2
        self.monsters = []

        for _ in range(num_monsters):
            monster = Ball('purple')
            self.place_obj(monster, top=(10, 1), size=(width - 11, height - 2))
            self.monsters.append(monster)

    def _move_monsters(self):
        """IA de los monstruos: Perseguir al agente"""
        if not hasattr(self, 'monsters'): return

        ax, ay = self.agent_pos

        for monster in self.monsters:
            if monster.cur_pos is None: continue  # Si por error se borró

            mx, my = monster.cur_pos

            # Lógica simple: Acercarse en el eje más lejano
            if abs(ax - mx) > abs(ay - my):
                dx = 1 if ax > mx else -1
                dy = 0
            else:
                dx = 0
                dy = 1 if ay > my else -1

            # Posible nueva posición
            new_x, new_y = mx + dx, my + dy

            # Verificar colisiones (No paredes, no lava, no otros monstruos)
            cell = self.grid.get(new_x, new_y)
            if cell is None or (cell.type == 'agent'):  # Puede moverse a vacío o comerse al agente
                self.grid.set(mx, my, None)  # Borrar rastro
                self.grid.set(new_x, new_y, monster)  # Mover
                monster.cur_pos = (new_x, new_y)

    def reset(self, *, seed=None, options=None):
        # Si es un reset "real" (muerte o inicio), reiniciamos ronda a 1
        # Si venimos de un "next_level", mantenemos la ronda
        if options and options.get('keep_progress'):
            pass  # Mantenemos self.round_count
        else:
            self.round_count = 1

        return super().reset(seed=seed, options=options)

    def step(self, action):
        # 1. Movimiento del Agente
        obs, reward, terminated, truncated, info = super().step(action)

        # 2. Movimiento de Monstruos
        self._move_monsters()

        # 3. Comprobar Muerte por Monstruo (Colisión)
        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell is not None and agent_cell.type == 'ball':
            reward = -1.0  # Muerte dolorosa
            terminated = True
            # print(f"💀 Murió en Ronda {self.round_count} devorado.")

        # 4. Comprobar "Meta" (Fin de Ronda)
        # Si el agente pisa la casilla Goal (o interactúa), NO termina el episodio.
        # En su lugar, REGENERAMOS el nivel.
        if terminated and reward > 0:  # Ha llegado a la meta
            # print(f"🚀 ¡Ronda {self.round_count} Superada! Aumentando dificultad...")

            # NO terminamos el episodio
            terminated = False

            # Reward por sobrevivir una ronda
            reward = 10.0 * self.round_count

            # Aumentar dificultad
            self.round_count += 1

            # Regenerar mapa sobre la marcha (Truco de magia)
            self._generate_level_obstacles()

            # Teletransportar agente al inicio
            self.place_agent(top=(1, 1), size=(2, self.grid_h - 2))

            # Importante: Actualizar la observación porque el mapa ha cambiado
            obs = self.gen_obs()

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-Endless-v1', entry_point='__main__:EndlessSurvivalEnv')
except:
    pass

# =========================================================
# ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-Endless-v1"
    models_dir = "models/Endless"
    log_dir = "logs_endless"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"--- ENDLESS RUNNER: Sobrevive infinitamente ---")

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    # Lógica de Resume
    model_path = f"{models_dir}/PPO_Endless_Latest.zip"
    if os.path.exists(model_path):
        print("Cargando cerebro previo...")
        model = PPO.load(model_path, env=vec_env, device="cpu")
    else:
        print("Cerebro nuevo...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.02,
            gamma=0.99,
            device="cpu",
            tensorboard_log=log_dir
        )

    # Entrenar
    # Como es infinito, el agente aprenderá a sobrevivir lo máximo posible
    total_timesteps = 5_000_000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)

    # VISUALIZACIÓN
    print("--- Visualizando Endless Mode ---")
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:  # Solo termina si MUERE
            print(">>> GAME OVER <<<")
            obs, _ = env.reset()
        elif reward > 1.0:  # Si recibe reward grande sin terminar, es que pasó de ronda
            print(">>> ¡NIVEL COMPLETADO! Siguiente ronda... <<<")