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
# 1. ENTORNO: SURVIVAL GAUNTLET (Lava + Monster)
# =========================================================
class SurvivalGauntletEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        # Mapa muy largo (32) y estrecho (9)
        self.grid_w = 32
        self.grid_h = 9

        mission_space = MissionSpace(mission_func=lambda: "survive and reach the end")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=1000,  # Pasos limitados para meter presión
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Dividimos en 4 Habitaciones de ancho 8
        # Room 1: 0-7 | Room 2: 8-15 | Room 3: 16-23 | Room 4: 24-31

        # Muros divisorios
        self.grid.vert_wall(7, 0, height)
        self.grid.vert_wall(15, 0, height)
        self.grid.vert_wall(23, 0, height)

        # --- PUERTAS (Locked) ---
        # Puerta 1
        self.door1 = Door('yellow', is_locked=True)
        self.grid.set(7, 4, self.door1)
        # Puerta 2
        self.door2 = Door('blue', is_locked=True)
        self.grid.set(15, 4, self.door2)
        # Puerta 3
        self.door3 = Door('red', is_locked=True)
        self.grid.set(23, 4, self.door3)

        # --- LLAVES (Una en cada habitación anterior) ---
        # Llave 1 (Yellow) en Room 1
        self.key1 = Key('yellow')
        self.place_obj(self.key1, top=(1, 1), size=(5, 7))

        # Llave 2 (Blue) en Room 2
        self.key2 = Key('blue')
        self.place_obj(self.key2, top=(8, 1), size=(6, 7))

        # Llave 3 (Red) en Room 3
        self.key3 = Key('red')
        self.place_obj(self.key3, top=(16, 1), size=(6, 7))

        # --- AMENAZAS ---

        # ROOM 2: LAVA STATIC
        # Patrón simple de lava para esquivar
        self.grid.set(10, 2, Lava())
        self.grid.set(10, 6, Lava())
        self.grid.set(11, 3, Lava())
        self.grid.set(11, 5, Lava())
        self.grid.set(12, 4, Lava())  # Justo en el centro, hay que rodear

        # ROOM 3: EL CAZADOR (Monster)
        # Usamos una 'Ball' morada como monstruo.
        # Guardamos la referencia para moverla en step()
        self.monster1 = Ball('purple')
        self.grid.set(20, 7, self.monster1)  # Empieza en la esquina

        # ROOM 4: CAOS (Lava + Monster 2)
        self.grid.set(26, 2, Lava())
        self.grid.set(26, 6, Lava())
        self.grid.set(28, 3, Lava())
        self.grid.set(28, 5, Lava())

        self.monster2 = Ball('purple')
        self.grid.set(29, 1, self.monster2)  # Otro monstruo

        # --- META ---
        self.place_obj(Goal(), top=(30, 1), size=(1, 7))

        # --- AGENTE ---
        self.place_agent(top=(1, 1), size=(3, 3))

    def _move_monster(self, monster):
        """Lógica simple de persecución: Se mueve hacia el agente"""
        if monster is None or monster.cur_pos is None: return

        mx, my = monster.cur_pos
        ax, ay = self.agent_pos

        # Decidir dirección (Horizontal primero)
        next_pos = None

        # Intentar moverse en X
        if mx < ax:
            dx = 1
        elif mx > ax:
            dx = -1
        else:
            dx = 0

        # Intentar moverse en Y
        if my < ay:
            dy = 1
        elif my > ay:
            dy = -1
        else:
            dy = 0

        # Prioridad aleatoria para que no se atasque
        if np.random.rand() < 0.5 and dx != 0:
            potential_pos = (mx + dx, my)
        elif dy != 0:
            potential_pos = (mx, my + dy)
        else:
            potential_pos = (mx + dx, my)  # Fallback

        # Comprobar si la casilla está libre (No paredes, no puertas cerradas)
        cell = self.grid.get(*potential_pos)

        # El monstruo puede moverse si está vacío o si es el agente (game over)
        can_move = (cell is None) or (cell is not None and cell.type == 'agent') or (
                    cell is not None and cell.type == 'door' and cell.is_open)

        if can_move:
            self.grid.set(mx, my, None)  # Borrar de la anterior
            self.grid.set(*potential_pos, monster)  # Poner en la nueva
            monster.cur_pos = potential_pos  # Actualizar ref interna

    def reset(self, *, seed=None, options=None):
        self.room_progress = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        pre_carrying = self.carrying

        # 1. MOVER AGENTE
        obs, reward, terminated, truncated, info = super().step(action)

        # 2. MOVER MONSTRUOS (Solo si no ha terminado el juego ya)
        if not terminated:
            # Mover monstruo 1 (Room 3)
            # Solo se mueve si el agente ha entrado en la Room 2 o superior (para dar tiempo)
            if self.agent_pos[0] > 7:
                self._move_monster(self.monster1)

            # Mover monstruo 2 (Room 4)
            if self.agent_pos[0] > 15:
                self._move_monster(self.monster2)

            # 3. COMPROBAR MUERTE POR MONSTRUO
            # Si la posición del agente coincide con una bola, muere.
            cell_agent = self.grid.get(*self.agent_pos)
            if cell_agent is not None and cell_agent.type == 'ball':
                reward = -1.0  # Castigo doloroso
                terminated = True
                # print(">>> ¡DEVORADO POR EL MONSTRUO! <<<")

        # --- REWARDS DE PROGRESO (Survival) ---
        # Recompensamos por cruzar umbrales X (avanzar habitaciones)
        agent_x = self.agent_pos[0]

        # Entrar en Room 2
        if agent_x > 7 and self.room_progress < 1:
            reward += 10.0
            self.room_progress = 1
            # print(">> Nivel 2: Lava")

        # Entrar en Room 3
        if agent_x > 15 and self.room_progress < 2:
            reward += 20.0
            self.room_progress = 2
            # print(">> Nivel 3: El Depredador")

        # Entrar en Room 4
        if agent_x > 23 and self.room_progress < 3:
            reward += 30.0
            self.room_progress = 3
            # print(">> Nivel 4: Infierno")

        # META
        if terminated and reward > 0:
            reward += 100.0

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-Gauntlet-v1', entry_point='__main__:SurvivalGauntletEnv')
except:
    pass

# =========================================================
# ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-Gauntlet-v1"

    # Directorios
    models_dir = "models/Gauntlet"
    log_dir = "logs_gauntlet"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"--- SURVIVAL GAUNTLET: Lava & Monsters ---")

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    # Resume Logic
    model_path = f"{models_dir}/PPO_Gauntlet.zip"

    if os.path.exists(model_path):
        print("Cargando modelo existente...")
        model = PPO.load(model_path, env=vec_env, device="cpu")
    else:
        print("Creando modelo nuevo...")
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

    # Entrenar (Se necesita tiempo para aprender a esquivar monstruos)
    total_timesteps = 4_000_000

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)

    # VISUALIZACIÓN
    print("--- Visualizando Supervivencia ---")
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)  # Necesario

    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            if reward > 0:
                print(">>> ¡SOBREVIVIÓ! <<<")
            else:
                print(">>> MUERTO (Lava o Monstruo) <<<")
            obs, _ = env.reset()