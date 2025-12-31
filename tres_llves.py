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
import os


# =========================================================
# 1. ENTORNO: HUB GLOBAL LAVA (DISTRIBUIDA Y POSIBLE)
# =========================================================
class HubGlobalLavaEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19

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

        # --- GEOMETRÍA CRUZ (CROSS) ---
        h_left, h_right = 6, 12
        h_top, h_bottom = 6, 12

        # Muros separadores (dejando los pasillos centrales libres)
        self.grid.vert_wall(h_left, 0, h_top)
        self.grid.horz_wall(0, h_top, h_left)

        self.grid.vert_wall(h_right, 0, h_top)
        self.grid.horz_wall(h_right, h_top, width - h_right)

        self.grid.vert_wall(h_left, h_bottom, height - h_bottom)
        self.grid.horz_wall(0, h_bottom, h_left)

        self.grid.vert_wall(h_right, h_bottom, height - h_bottom)
        self.grid.horz_wall(h_right, h_bottom, width - h_right)

        # Caja del Hub
        self.grid.wall_rect(h_left, h_top, 7, 7)

        # --- PUERTAS ---
        self.door_red = Door('red', is_locked=True)
        self.grid.set(9, h_top, self.door_red)  # Norte

        self.door_blue = Door('blue', is_locked=True)
        self.grid.set(h_right, 9, self.door_blue)  # Este

        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(h_left, 9, self.door_yellow)  # Oeste

        # --- OBJETOS CLAVE ---

        # 1. Llave Roja (Hub Centro)
        self.key_red = Key('red')
        self.grid.set(9, 9, self.key_red)

        # 2. Llave Azul (Sala Norte - Fondo)
        self.key_blue = Key('blue')
        self.grid.set(9, 1, self.key_blue)

        # 3. Llave Amarilla (Sala Este - Fondo)
        self.key_yellow = Key('yellow')
        self.grid.set(17, 9, self.key_yellow)

        # 4. Meta (Sala Oeste - Fondo)
        self.place_obj(Goal(), top=(1, 9), size=(1, 1))

        # --- DISTRIBUCIÓN DE LAVA (EL RETO) ---
        # Colocamos lava asegurándonos de NO bloquear el camino principal.

        # 1. SALA NORTE (El Pasillo Estrecho)
        # Dejamos libre solo la columna central (x=9). Llenamos los lados.
        for y in range(1, h_top):
            self.grid.set(7, y, Lava())  # Lado izquierdo
            self.grid.set(8, y, Lava())
            self.grid.set(10, y, Lava())  # Lado derecho
            self.grid.set(11, y, Lava())

        # 2. SALA ESTE (Islas Alternas)
        # Ponemos lava arriba y abajo, dejando el camino central (y=9) libre.
        for x in range(h_right + 1, width - 1):
            self.grid.set(x, 7, Lava())
            self.grid.set(x, 11, Lava())
            # Un poco de zig-zag extra
            if x % 2 == 0:
                self.grid.set(x, 8, Lava())
                self.grid.set(x, 10, Lava())

        # 3. SALA OESTE (El campo de minas final)
        # Protegemos la meta con lava alrededor, pero dejando acceso.
        # Meta está en (1,9) aprox.
        self.grid.set(2, 8, Lava())
        self.grid.set(2, 10, Lava())
        self.grid.set(3, 9, Lava())  # Hay que rodear este bloque

        # Relleno extra en las esquinas de la sala oeste
        for x in range(1, h_left):
            self.grid.set(x, 6, Lava())  # Pegado al muro norte
            self.grid.set(x, 12, Lava())  # Pegado al muro sur

        # 4. HUB CENTRAL (Esquinas peligrosas)
        # Para que no se acomode, ponemos lava en las esquinas del Hub.
        self.grid.set(h_left + 1, h_top + 1, Lava())
        self.grid.set(h_right - 1, h_top + 1, Lava())
        self.grid.set(h_left + 1, h_bottom - 1, Lava())
        self.grid.set(h_right - 1, h_bottom - 1, Lava())

        # 5. SALA SUR (Decorativa pero peligrosa)
        for x in range(h_left + 1, h_right):
            for y in range(h_bottom + 1, height - 1):
                if (x + y) % 2 == 0:  # Patrón de ajedrez
                    self.grid.set(x, y, Lava())

        # --- AGENTE ---
        # Lo ponemos en una casilla segura del Hub (cerca de la puerta sur, pero dentro)
        self.place_agent(top=(8, 8), size=(3, 3))

    def reset(self, *, seed=None, options=None):
        self.has_red = False
        self.opened_red = False
        self.has_blue = False
        self.opened_blue = False
        self.has_yellow = False
        self.opened_yellow = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        pre_carrying = self.carrying
        pre_d_red = self.door_red.is_open
        pre_d_blue = self.door_blue.is_open
        pre_d_yellow = self.door_yellow.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # REWARDS ESCALONADOS

        # 1. Coger Roja
        if not self.has_red and pre_carrying != self.key_red and self.carrying == self.key_red:
            reward += 5.0
            self.has_red = True

        # 2. Abrir Norte
        elif self.has_red and not self.opened_red and not pre_d_red and self.door_red.is_open:
            reward += 10.0
            self.opened_red = True

        # 3. Coger Azul (Swap)
        elif self.opened_red and not self.has_blue and pre_carrying != self.key_blue and self.carrying == self.key_blue:
            reward += 15.0
            self.has_blue = True

        # 4. Abrir Este
        elif self.has_blue and not self.opened_blue and not pre_d_blue and self.door_blue.is_open:
            reward += 20.0
            self.opened_blue = True

        # 5. Coger Amarilla (Swap)
        elif self.opened_blue and not self.has_yellow and pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            reward += 25.0
            self.has_yellow = True

        # 6. Abrir Oeste (Meta)
        elif self.has_yellow and not self.opened_yellow and not pre_d_yellow and self.door_yellow.is_open:
            reward += 30.0
            self.opened_yellow = True

        # Meta
        if terminated and reward > 0:
            reward += 100.0

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-HubLavaGlobal-v4', entry_point='__main__:HubGlobalLavaEnv')
except:
    pass

# =========================================================
# ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-HubLavaGlobal-v4"
    log_path = "HubLavaLogs"
    os.makedirs(log_path, exist_ok=True)

    print(f"--- Entrenando 'HUB LAVA GLOBAL' (Mapa Complejo) ---")

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

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

    # 4 Millones de pasos necesarios para maniobrar con cuidado
    total_timesteps = 4_000_000

    model.learn(total_timesteps=total_timesteps)

    save_path = "PPO_HubLavaGlobal"
    model.save(save_path)

    # VISUALIZACIÓN
    print("--- Visualizando ---")
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(save_path, device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            if reward > 0:
                print(">>> ¡ÉXITO! <<<")
            else:
                print(">>> Murió en la lava <<<")
            obs, _ = env.reset()