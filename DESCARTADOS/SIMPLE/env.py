import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
import os

class CustomDoorKeyEnv(MiniGridEnv):
    def __init__(self, size=10, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = "Coge la llave, abre la puerta y llega a la meta"
        mission_space = MissionSpace(mission_func=lambda: self.mission)
        
        # Variable para controlar si ya le dimos el premio por la llave
        self.rewarded_for_key = False 

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size**2,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Pared vertical
        splitIdx = width // 2
        for i in range(0, height):
            self.grid.set(splitIdx, i, Wall())

        # Puerta y Llave
        doorIdx = height // 2
        self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))
        self.grid.set(2, 2, Key('yellow'))
        self.grid.set(width - 2, height - 2, Goal())

        # Agente
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        # Reiniciamos el control del premio
        self.rewarded_for_key = False

    def step(self, action):
        # Ejecutamos el paso normal de Minigrid
        obs, reward, terminated, truncated, info = super().step(action)
        
        # --- REWARD SHAPING (El Truco) ---
        # Si el agente lleva algo (self.carrying) Y es la llave Y no le hemos pagado aún:
        if self.carrying and self.carrying.type == 'key' and not self.rewarded_for_key:
            reward += 0.5        # ¡Premio!
            self.rewarded_for_key = True # Marcamos para no pagarle doble
            
        return obs, reward, terminated, truncated, info

# --- Entrenamiento ---
if __name__ == "__main__":
    # 1. Crear entorno
    env = CustomDoorKeyEnv(render_mode="rgb_array", size=10)
    env = FlatObsWrapper(env) # Aplanar la visión

    # 2. Modelo (Forzamos CPU para velocidad)
    print("Iniciando entrenamiento TURBO...")
    # device='cpu' elimina el cuello de botella de la GPU en redes pequeñas
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device='cpu')

    # 3. Entrenar (Subimos un poco los pasos, pero aprenderá antes)
    model.learn(total_timesteps=150000)
    print("¡Entrenamiento finalizado!")

    # 4. Guardar y Probar
    model.save("ppo_doorkey_smart")
    
    # Visualización
    print("Visualizando...")
    eval_env = CustomDoorKeyEnv(render_mode="human", size=10)
    eval_env = FlatObsWrapper(eval_env)
    
    obs, _ = eval_env.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = eval_env.step(action)
        eval_env.render()
        if term or trunc:
            obs, _ = eval_env.reset()