import gymnasium as gym
import time
import os
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN


# ==========================================
# CLASE DEL ENTORNO (Debe ser IDÉNTICA a la de entrenar)
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
        return super().step(action)


# ==========================================
# MAIN PRUEBA
# ==========================================
ACTION_NAMES = {0: "Left", 1: "Right", 2: "Forward", 3: "Pickup", 4: "Drop", 5: "Toggle", 6: "Done"}


def main():
    model_path = "models/portal_final/modelo_terminado.zip"

    if not os.path.exists(model_path):
        print(f"¡ERROR! No encuentro: {model_path}")
        print("Ejecuta 'entrenar.py' primero.")
        return

    print(">>> CARGANDO AGENTE... (Ctrl+C para salir)")
    model = DQN.load(model_path)

    env = PortalChamberEnv(render_mode="human", size=8)
    env = FlatObsWrapper(env)

    obs, _ = env.reset()

    try:
        while True:
            # Deterministic=True: Usa lo mejor que sabe (sin inventar)
            action, _ = model.predict(obs, deterministic=True)

            # Debug en consola
            print(f"Acción: {ACTION_NAMES.get(int(action), 'Unknown')}")

            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            time.sleep(0.1)

            if terminated or truncated:
                if reward > 0:
                    print(">>> ¡NIVEL COMPLETADO! 🎉")
                else:
                    print("--- Fallo ---")
                obs, _ = env.reset()
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nCerrando...")
        env.close()


if __name__ == "__main__":
    main()