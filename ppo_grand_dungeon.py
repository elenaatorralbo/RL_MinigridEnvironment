import gymnasium as gym
import minigrid
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Ball
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


class SimpleMovementWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, act):
        return act


class GrandDungeonLogicEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 25
        self.grid_h = 25
        mission_space = MissionSpace(mission_func=lambda: "conquer the four realms")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=8000,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        mid_x = width // 2
        mid_y = height // 2

        # Muros divisorios
        self.grid.vert_wall(mid_x, 0, height)
        self.grid.horz_wall(0, mid_y, width)

        # PUERTAS (Ahora son vitales)
        self.door_red = Door('red', is_locked=True);
        self.grid.set(mid_x, 6, self.door_red)
        # La puerta azul está en (18, 12). El agente debe llegar aquí desde arriba.
        self.door_blue = Door('blue', is_locked=True);
        self.grid.set(18, mid_y, self.door_blue)
        self.door_yellow = Door('yellow', is_locked=True);
        self.grid.set(mid_x, 18, self.door_yellow)

        # --- ZONA 1: RUINAS ---
        for y in range(2, mid_y - 2):
            if y % 2 == 0: self.grid.set(6, y, Wall())  # Pilares alternos
        self.pos_key_red = (2, 2)
        self.grid.set(*self.pos_key_red, Key('red'))

        # --- ZONA 2: LAVA (CAMINO DE IDA Y VUELTA) ---
        # 1. Llenar de lava
        for x in range(mid_x + 1, width - 1):
            for y in range(1, mid_y):
                self.grid.set(x, y, Lava())

        # 2. ESCULPIR CAMINOS (Vaciando la lava)

        # A) Conexión Puerta Roja (Entrada)
        for y in range(1, 7): self.grid.set(mid_x + 1, y, None)

        # B) Autopista Superior (Hacia la llave)
        for x in range(mid_x + 1, width - 1): self.grid.set(x, 2, None)

        # C) Bajada Lateral Derecha
        for y in range(2, mid_y): self.grid.set(width - 2, y, None)

        # D) CAMINO DE RETORNO A LA PUERTA AZUL (La clave que faltaba)
        # La puerta azul está en (18, 12).
        # Limpiamos la fila 11 (mid_y - 1) desde la derecha hasta la puerta
        for x in range(18, width - 1):
            self.grid.set(x, mid_y - 1, None)

        self.pos_key_blue = (width - 2, 2)
        self.grid.set(*self.pos_key_blue, Key('blue'))

        # --- ZONA 3: ARENA ---
        self.grid.set(mid_x + 4, mid_y + 4, Wall())
        self.grid.set(mid_x + 8, mid_y + 4, Wall())
        self.grid.set(mid_x + 4, mid_y + 8, Wall())
        self.grid.set(mid_x + 8, mid_y + 8, Wall())

        self.monsters = []
        try:
            m1 = Ball('purple');
            self.grid.set(width - 3, height - 3, m1);
            self.monsters.append(m1)
            m2 = Ball('purple');
            self.grid.set(mid_x + 2, height - 5, m2);
            self.monsters.append(m2)
        except:
            pass

        self.pos_key_yellow = (width - 2, height - 2)
        self.grid.set(*self.pos_key_yellow, Key('yellow'))

        # --- ZONA 4: ESPIRAL ---
        for x in range(2, mid_x - 2): self.grid.set(x, mid_y + 2, Lava())
        for y in range(mid_y + 2, height - 2): self.grid.set(2, y, Lava())
        for x in range(2, mid_x - 2): self.grid.set(x, height - 3, Lava())
        for y in range(mid_y + 4, height - 3): self.grid.set(mid_x - 3, y, Lava())

        self.pos_goal = (mid_x // 2, height // 2 + (height // 4))
        self.place_obj(Goal(), top=self.pos_goal, size=(1, 1))

        # --- SETUP ---
        self.agent_pos = (mid_x - 2, mid_y - 2)
        self.agent_dir = 3
        self.has_red = False;
        self.opened_red = False
        self.has_blue = False;
        self.opened_blue = False
        self.has_yellow = False;
        self.opened_yellow = False

        # Checkpoints flags (para no repetir premios)
        self.zone2_accessed = False
        self.zone3_accessed = False
        self.zone4_accessed = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.zone2_accessed = False
        self.zone3_accessed = False
        self.zone4_accessed = False

        # Migas de pan (Breadcrumbs) para ayudar al principio
        self.breadcrumbs = []
        # Diagonal hacia llave roja
        for i in range(1, 9): self.breadcrumbs.append((self.agent_pos[0] - i, self.agent_pos[1] - i))

        self.target_pos = self._get_target_pos()
        self.prev_dist = self._get_dist_to(self.target_pos)
        return obs, info

    def _get_target_pos(self):
        # Lógica secuencial estricta
        if not self.has_red: return np.array(self.pos_key_red)
        if not self.opened_red: return np.array((self.grid_w // 2, 6))  # Ir a Puerta Roja

        if not self.has_blue: return np.array(self.pos_key_blue)
        # AQUÍ ESTÁ EL CAMBIO: Si tienes la llave azul, tu objetivo es la PUERTA AZUL
        if not self.opened_blue: return np.array((18, self.grid_h // 2))

        if not self.has_yellow: return np.array(self.pos_key_yellow)
        if not self.opened_yellow: return np.array((self.grid_w // 2, 18))

        return np.array(self.pos_goal)

    def _get_dist_to(self, target):
        return np.sum(np.abs(np.array(self.agent_pos) - target))

    def dist_between(self, pos_a, pos_b):
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def _move_monsters(self):
        mid_x, mid_y = self.grid_w // 2, self.grid_h // 2
        if self.agent_pos[0] > mid_x and self.agent_pos[1] > mid_y:
            for monster in self.monsters:
                if monster.cur_pos is None: continue
                mx, my = monster.cur_pos
                ax, ay = self.agent_pos
                best_move = None
                min_dist = abs(mx - ax) + abs(my - ay)
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                np.random.shuffle(moves)
                for dx, dy in moves:
                    nx, ny = mx + dx, my + dy
                    cell = self.grid.get(nx, ny)
                    if cell is None or (cell.type == 'agent'):
                        dist = abs(nx - ax) + abs(ny - ay)
                        if dist < min_dist: min_dist = dist; best_move = (nx, ny)
                if best_move:
                    self.grid.set(mx, my, None);
                    self.grid.set(best_move[0], best_move[1], monster)
                    monster.cur_pos = best_move

    def step(self, action):
        self._move_monsters()
        obs, reward, terminated, truncated, info = super().step(action)
        state_changed = False

        # --- 1. COGER LLAVES (Sin Teleport) ---
        if not self.has_red and self.dist_between(self.agent_pos, self.pos_key_red) <= 1:
            self.has_red = True;
            self.grid.set(*self.pos_key_red, None)
            reward += 30.0;
            state_changed = True
            print(">> LLAVE ROJA: Obtenida! Ve a la puerta.")

        elif not self.has_blue and self.dist_between(self.agent_pos, self.pos_key_blue) <= 1:
            self.has_blue = True;
            self.grid.set(*self.pos_key_blue, None)
            reward += 30.0;
            state_changed = True
            print(">> LLAVE AZUL: Obtenida! Vuelve a la puerta azul.")

        elif not self.has_yellow and self.dist_between(self.agent_pos, self.pos_key_yellow) <= 1:
            self.has_yellow = True;
            self.grid.set(*self.pos_key_yellow, None)
            reward += 30.0;
            state_changed = True
            print(">> LLAVE AMARILLA: Obtenida!")

        # --- 2. ABRIR PUERTAS Y CRUZAR (Checkpoints reales) ---
        front_cell = self.grid.get(*self.front_pos)

        # Abrir puertas
        if action == self.actions.forward and front_cell and front_cell.type == 'door':
            if front_cell.color == 'red' and self.has_red:
                self.door_red.is_open = True;
                self.opened_red = True;
                reward += 20;
                state_changed = True
            elif front_cell.color == 'blue' and self.has_blue:
                self.door_blue.is_open = True;
                self.opened_blue = True;
                reward += 20;
                state_changed = True
            elif front_cell.color == 'yellow' and self.has_yellow:
                self.door_yellow.is_open = True;
                self.opened_yellow = True;
                reward += 20;
                state_changed = True

        # DETECTAR SI HA CRUZADO A LA SIGUIENTE ZONA
        # Zona 2 (Lava): x > 12, y < 12
        if not self.zone2_accessed and self.agent_pos[0] > 12 and self.agent_pos[1] < 12:
            self.zone2_accessed = True
            reward += 50.0
            print(">>> ENTRANDO EN ZONA DE LAVA")

        # Zona 3 (Arena): x > 12, y > 12 (Tras cruzar puerta azul)
        if not self.zone3_accessed and self.agent_pos[0] > 12 and self.agent_pos[1] > 12:
            self.zone3_accessed = True
            reward += 50.0
            # Aquí sí podemos hacer un teleport pequeño para centrarlo y que no le coman los monstruos al entrar
            self.agent_pos = (15, 15);
            self.agent_dir = 1
            print(">>> ENTRANDO EN LA ARENA")

        # Zona 4 (Espiral): x < 12, y > 12
        if not self.zone4_accessed and self.agent_pos[0] < 12 and self.agent_pos[1] > 12:
            self.zone4_accessed = True
            reward += 50.0
            print(">>> ENTRANDO EN LA ESPIRAL")

        # Migas de pan
        if not self.has_red and self.agent_pos in self.breadcrumbs:
            reward += 1.0;
            self.breadcrumbs.remove(self.agent_pos)

        # Muerte
        cell_agent = self.grid.get(*self.agent_pos)
        if cell_agent is not None and cell_agent.type == 'ball': reward = -50.0; terminated = True; print(
            ">>> MUERTO <<<")

        if terminated and reward > 0: reward += 500.0
        reward -= 0.005

        self.target_pos = self._get_target_pos()
        dist_now = self._get_dist_to(self.target_pos)
        if state_changed:
            self.prev_dist = dist_now
        else:
            reward += (self.prev_dist - dist_now) * 2.0; self.prev_dist = dist_now

        return obs, reward, terminated, truncated, info


try:
    register(id='MiniGrid-GrandDungeonLogic-v10', entry_point='__main__:GrandDungeonLogicEnv')
except:
    pass

if __name__ == "__main__":
    env_id = "MiniGrid-GrandDungeonLogic-v10"
    TOTAL_TIMESTEPS = 2_000_000
    vec_env = make_vec_env(env_id, n_envs=8, wrapper_class=lambda e: FlatObsWrapper(SimpleMovementWrapper(e)))
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01,
                gamma=0.995, device="cpu")


    class ProgressBar(BaseCallback):
        def _on_training_start(self): self.pbar = tqdm(total=self.locals['total_timesteps'])

        def _on_step(self): self.pbar.update(self.training_env.num_envs); return True

        def _on_training_end(self): self.pbar.close()


    print(f"🚀 Iniciando TFG LÓGICO (Uso real de puertas)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressBar(TOTAL_TIMESTEPS))
    model.save("ppo_grand_dungeon_logic")

    print("--- Demo ---")
    env = gym.make(env_id, render_mode="human")
    env = SimpleMovementWrapper(env)
    env = FlatObsWrapper(env)
    model = PPO.load("ppo_grand_dungeon_logic", device="cpu")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated: obs, _ = env.reset()