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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
from tqdm.auto import tqdm  # Barra de carga


# =========================================================
# 1. UTILIDAD: BARRA DE PROGRESO
# =========================================================
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Sobreviviendo", unit="steps")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


# =========================================================
# 2. ENTORNO: ENDLESS RUNNER (CORREGIDO)
# =========================================================
class EndlessSurvivalEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 20
        self.grid_h = 9

        mission_space = MissionSpace(mission_func=lambda: "survive as many rounds as possible")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=5000,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        # Inicializamos ronda si no existe
        if not hasattr(self, 'round_count'):
            self.round_count = 1

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Generar nivel
        self._generate_level_obstacles()

        # Agente
        self.place_agent(top=(1, 1), size=(2, height - 2))

        # Meta
        self.grid.set(width - 2, height // 2, Goal())

    def _generate_level_obstacles(self):
        """Genera obstáculos procedimentales"""
        width, height = self.grid.width, self.grid.height

        # 1. Limpiar zona interior
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                self.grid.set(i, j, None)

        # 2. Reponer Meta
        self.grid.set(width - 2, height // 2, Goal())

        # 3. Dificultad
        difficulty = self.round_count

        # LAVA
        num_lava = int(difficulty * 2)
        # Intentamos poner lava, si no hay sitio no pasa nada (try/except implícito en place_obj)
        for _ in range(num_lava):
            try:
                self.place_obj(Lava(), top=(3, 1), size=(width - 5, height - 2), max_tries=10)
            except:
                pass

                # MONSTRUOS
        # Reiniciamos la lista SIEMPRE para no mover monstruos borrados
        self.monsters = []

        num_monsters = difficulty // 2
        for _ in range(num_monsters):
            try:
                monster = Ball('purple')
                self.place_obj(monster, top=(10, 1), size=(width - 11, height - 2), max_tries=10)
                self.monsters.append(monster)
            except:
                pass

    def _move_monsters(self):
        """IA Monstruos (CORREGIDA: Comprobación de límites)"""
        if not hasattr(self, 'monsters'): return

        ax, ay = self.agent_pos

        for monster in self.monsters:
            if monster.cur_pos is None: continue

            mx, my = monster.cur_pos

            # Cálculo de dirección
            if abs(ax - mx) > abs(ay - my):
                dx = 1 if ax > mx else -1
                dy = 0
            else:
                dx = 0
                dy = 1 if ay > my else -1

            new_x, new_y = mx + dx, my + dy

            # --- CORRECCIÓN DEL BUG ---
            # Verificamos que la nueva posición esté DENTRO del array antes de mirar qué hay
            if not (0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height):
                continue  # Si se sale del mapa, abortar movimiento

            # Ahora es seguro mirar
            cell = self.grid.get(new_x, new_y)

            # Se mueve si está vacío o si es el agente (para matarlo)
            if cell is None or (cell.type == 'agent'):
                self.grid.set(mx, my, None)
                self.grid.set(new_x, new_y, monster)
                monster.cur_pos = (new_x, new_y)

    def reset(self, *, seed=None, options=None):
        if options and options.get('keep_progress'):
            pass
        else:
            self.round_count = 1

        return super().reset(seed=seed, options=options)

    def step(self, action):
        # 1. Agente
        obs, reward, terminated, truncated, info = super().step(action)

        # 2. Monstruos
        self._move_monsters()

        # 3. Muerte
        # Verificamos colisión post-movimiento
        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell is not None and agent_cell.type == 'ball':
            reward = -1.0
            terminated = True

        # 4. Siguiente Ronda (Meta)
        if terminated and reward > 0:
            terminated = False  # ¡No acaba el episodio!
            reward = 10.0 * self.round_count  # Puntos por ronda

            self.round_count += 1

            # Regenerar mapa
            self._generate_level_obstacles()

            # Teleport Agente
            self.place_agent(top=(1, 1), size=(2, self.grid_h - 2))

            # Actualizar observación
            obs = self.gen_obs()

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-Endless-Fixed-v2', entry_point='__main__:EndlessSurvivalEnv')
except:
    pass

# =========================================================
# 3. ENTRENAMIENTO
# =========================================================

if __name__ == "__main__":
    env_id = "MiniGrid-Endless-Fixed-v2"

    # Carpetas
    models_dir = "models/Endless"
    log_dir = "logs_endless"
    checkpoint_dir = "checkpoints_endless"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"--- ENDLESS RUNNER (FIXED): Sobrevive ---")

    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=FlatObsWrapper)

    model_path = f"{models_dir}/PPO_Endless_Latest.zip"

    if os.path.exists(model_path):
        print("✅ Cargando cerebro previo...")
        model = PPO.load(model_path, env=vec_env, device="cpu")
        reset_steps = False
    else:
        print("🆕 Cerebro nuevo...")
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
        reset_steps = True

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=checkpoint_dir,
        name_prefix="ppo_endless"
    )
    bar_callback = ProgressBarCallback(5_000_000)

    # Entrenar
    print("🚀 Iniciando entrenamiento...")
    model.learn(
        total_timesteps=5_000_000,
        callback=[checkpoint_callback, bar_callback],
        progress_bar=False,
        reset_num_timesteps=reset_steps
    )

    model.save(model_path)
    print("🏁 Guardado.")

    # VISUALIZACIÓN
    print("--- Visualizando ---")
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            print(">>> MUERTO <<<")
            obs, _ = env.reset()
        elif reward > 1.0:
            print(">>> ¡RONDA SUPERADA! <<<")