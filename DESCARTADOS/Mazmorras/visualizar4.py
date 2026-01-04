import gymnasium as gym
import numpy as np
import os
import time
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from sb3_contrib import RecurrentPPO

# ==========================================
# CONFIGURACIÓN DE PRUEBA
# ==========================================
# CAMBIA ESTO: 1, 2 o 3 para ver las distintas fases
NIVEL_A_PROBAR = 2
# ==========================================

class SpiralTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 11
        self.grid_h = 11
        self.current_level = NIVEL_A_PROBAR  # Usamos la variable global
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the spiral temple")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=1000,
            render_mode=render_mode
        )

    def set_level(self, level):
        self.current_level = level

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # PAREDES (Cruz central)
        self.grid.vert_wall(5, 0)
        self.grid.horz_wall(0, 5)

        # PUERTAS Y HUECOS
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(5, 2, self.door_yellow)  # Puerta Amarilla
        
        self.door_red = Door('red', is_locked=True)
        self.grid.set(8, 5, self.door_red)     # Puerta Roja
        
        self.grid.set(5, 8, None)              # HUECO (Gap) abajo

        # OBJETOS
        self.key_yellow = Key('yellow')
        self.place_obj(self.key_yellow, top=(1, 1), size=(4, 4))
        self.key_red = Key('red')
        self.place_obj(self.key_red, top=(6, 1), size=(4, 4))

        # META (Abajo-Izquierda)
        self.place_obj(Goal(), top=(1, 6), size=(4, 4))

        # Lava decorativa
        self.grid.set(7, 7, Lava())

        # --- SPAWN LOGIC (LO QUE QUEREMOS PROBAR) ---
        if self.current_level == 1:
            # Fase 1: En la meta (Abajo-Izquierda)
            self.place_agent(top=(1, 6), size=(4, 4))
        
        elif self.current_level == 2:
            # Fase 2: Arriba-Derecha (Llave Roja -> Puerta Roja -> Hueco -> Meta)
            self.place_agent(top=(6, 1), size=(4, 4))
            self.door_yellow.is_open = True 
            
        else: # Nivel 3
            # Fase 3: Arriba-Izquierda (Todo el recorrido)
            self.place_agent(top=(1, 1), size=(4, 4))

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        return super().step(action)

# Registrar
if 'MiniGrid-Spiral-v0' in gym.envs.registry:
    del gym.envs.registry['MiniGrid-Spiral-v0']
register(id='MiniGrid-Spiral-v0', entry_point='__main__:SpiralTempleEnv')

def main():
    # Intenta cargar el mejor modelo, si no existe, carga el final
    model_path = "./SpiralTemple/Models/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "./SpiralTemple/Models/Final_Spiral_Agent.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ No encuentro modelo en {model_path}")
        print("¿Has ejecutado 'entrenar_espiral.py' primero?")
        return

    print(f"🌀 Auditando Nivel {NIVEL_A_PROBAR} | Modelo: {model_path}")

    env = gym.make('MiniGrid-Spiral-v0', render_mode='human')
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)

    model = RecurrentPPO.load(model_path, env=env)

    num_episodes = 5
    for i in range(num_episodes):
        print(f"--- Episodio {i+1} ---")
        obs, _ = env.reset()
        done = False
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            env.render()
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_starts = done
            time.sleep(0.05) # Velocidad de visualización

    env.close()

if __name__ == "__main__":
    main()