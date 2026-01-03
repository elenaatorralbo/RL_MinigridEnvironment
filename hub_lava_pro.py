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
from stable_baselines3.common.callbacks import CheckpointCallback
import os


# =========================================================
# 1. ENTORNO: HUB LAVA (CORREGIDO Y ENSANCHADO)
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

        # --- GEOMETRÍA CRUZ ---
        h_left, h_right = 6, 12
        h_top, h_bottom = 6, 12

        # Muros
        self.grid.vert_wall(h_left, 0, h_top)
        self.grid.horz_wall(0, h_top, h_left)
        self.grid.vert_wall(h_right, 0, h_top)
        self.grid.horz_wall(h_right, h_top, width - h_right)
        self.grid.vert_wall(h_left, h_bottom, height - h_bottom)
        self.grid.horz_wall(0, h_bottom, h_left)
        self.grid.vert_wall(h_right, h_bottom, height - h_bottom)
        self.grid.horz_wall(h_right, h_bottom, width - h_right)
        self.grid.wall_rect(h_left, h_top, 7, 7)  # Caja Hub

        # --- PUERTAS ---
        self.door_red = Door('red', is_locked=True)
        self.grid.set(9, h_top, self.door_red)  # Norte
        self.door_blue = Door('blue', is_locked=True)
        self.grid.set(h_right, 9, self.door_blue)  # Este
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(h_left, 9, self.door_yellow)  # Oeste

        # --- OBJETOS ---
        self.key_red = Key('red')
        self.grid.set(9, 9, self.key_red)
        self.key_blue = Key('blue')
        self.grid.set(9, 1, self.key_blue)
        self.key_yellow = Key('yellow')
        self.grid.set(17, 9, self.key_yellow)
        self.place_obj(Goal(), top=(1, 9), size=(1, 1))

        # --- LAVA (DISTRIBUCIÓN MEJORADA) ---

        # 1. SALA NORTE (Arreglado: Pasillo Ancho)
        # Dejamos x=8, 9, 10 libres para que pueda dar la vuelta sin quemarse
        for y in range(1, h_top):
            self.grid.set(7, y, Lava())
            self.grid.set(11, y, Lava())

            # 2. SALA ESTE (Islas)
        for x in range(h_right + 1, width - 1):
            self.grid.set(x, 7, Lava())
            self.grid.set(x, 11, Lava())
            if x % 2 == 0:
                self.grid.set(x, 8, Lava())
                self.grid.set(x, 10, Lava())

        # 3. SALA OESTE (Rodeo Meta)
        self.grid.set(2, 8, Lava())
        self.grid.set(2, 10, Lava())
        self.grid.set(3, 9, Lava())
        for x in range(1, h_left):
            self.grid.set(x, 6, Lava())
            self.grid.set(x, 12, Lava())

        # 4. HUB CENTRAL (Esquinas)
        self.grid.set(h_left + 1, h_top + 1, Lava())
        self.grid.set(h_right - 1, h_top + 1, Lava())
        self.grid.set(h_left + 1, h_bottom - 1, Lava())
        self.grid.set(h_right - 1, h_bottom - 1, Lava())

        # 5. SALA SUR (Decorativa)
        for x in range(h_left + 1, h_right):
            for y in range(h_bottom + 1, height - 1):
                if (x + y) % 2 == 0:
                    self.grid.set(x, y, Lava())

        # --- AGENTE ---
        self.place_agent(top=(8, 8), size=(3, 3))

    def reset(self, *, seed=None, options=None):
        self.has_red = False;
        self.opened_red = False
        self.has_blue = False;
        self.opened_blue = False
        self.has_yellow = False;
        self.opened_yellow = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        pre_carrying = self.carrying
        pre_d_red = self.door_red.is_open
        pre_d_blue = self.door_blue.is_open
        pre_d_yellow = self.door_yellow.is_open
        obs, reward, terminated, truncated, info = super().step(action)

        # REWARDS
        if not self.has_red and pre_carrying != self.key_red and self.carrying == self.key_red:
            reward += 5.0;
            self.has_red = True
        elif self.has_red and not self.opened_red and not pre_d_red and self.door_red.is_open:
            reward += 10.0;
            self.opened_red = True
        elif self.opened_red and not self.has_blue and pre_carrying != self.key_blue and self.carrying == self.key_blue:
            reward += 15.0;
            self.has_blue = True
        elif self.has_blue and not self.opened_blue and not pre_d_blue and self.door_blue.is_open:
            reward += 20.0;
            self.opened_blue = True
        elif self.opened_blue and not self.has_yellow and pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            reward += 25.0;
            self.has_yellow = True
        elif self.has_yellow and not self.opened_yellow and not pre_d_yellow and self.door_yellow.is_open:
            reward += 30.0;
            self.opened_yellow = True

        if terminated and reward > 0: reward += 100.0
        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-HubLavaPro-v5', entry_point='__main__:HubGlobalLavaEnv')
except:
    pass

# =========================================================
# 2. SISTEMA DE ENTRENAMIENTO PRO (RESUME + CHECKPOINTS)
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-HubLavaPro-v5"

    # Carpetas
    models_dir = "models/PPO"
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"--- Sistema PRO: {env_id} ---")

    # Entornos paralelos
    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    # Nombre del archivo final
    model_name = "PPO_HubLava_Latest"
    model_path = f"{models_dir}/{model_name}.zip"

    # --- LÓGICA DE RESUME (Reanudar) ---
    if os.path.exists(model_path):
        print(f"✅ ¡Modelo encontrado! Cargando '{model_name}'...")
        # reset_num_timesteps=False hace que la gráfica siga donde se quedó, no desde 0
        model = PPO.load(model_path, env=vec_env, device="cpu")
        reset_timesteps = False
        print(">> Reanudando entrenamiento...")
    else:
        print("🆕 No hay modelo guardado. Creando uno nuevo desde CERO.")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.035,  # Entropía alta para explorar
            gamma=0.99,
            device="cpu",
            tensorboard_log=log_dir
        )
        reset_timesteps = True

    # --- CHECKPOINTS (Guardado periódico) ---
    # Guarda cada 100,000 pasos (aprox) en la carpeta 'checkpoints'
    checkpoint_callback = CheckpointCallback(
        save_freq=12500,  # 12500 * 8 envs = 100,000 pasos reales
        save_path=checkpoint_dir,
        name_prefix="ppo_hub_lava"
    )

    # --- ENTRENAMIENTO CON BARRA DE PROGRESO ---
    total_timesteps = 5_000_000

    print(f"🚀 Iniciando entrenamiento. Objetivo: {total_timesteps} pasos.")
    # progress_bar=True muestra la cuenta atrás visual
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
        progress_bar=True
    )

    # Guardado final
    model.save(model_path)
    print("🏁 Entrenamiento finalizado y guardado.")

    # =========================================================
    # 3. VISUALIZACIÓN
    # =========================================================
    print("--- Visualizando ---")
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            if reward > 0:
                print(">>> ¡ÉXITO! <<<")
            else:
                print(">>> Fallo <<<")
            obs, _ = env.reset()