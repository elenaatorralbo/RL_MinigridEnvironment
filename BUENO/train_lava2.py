import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.core.grid import Grid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import os
import collections
import numpy as np

# =============================================================================
# 1. CUSTOM MULTICOLOR CORRIDOR WITH SMART LAVA ENVIRONMENT
# =============================================================================
class CorredorLavaSmart(MultiRoomEnv):
    def __init__(self, n_rooms=4, lava_prob=0.1, key_prob=0.1, **kwargs):
        # Guardamos la probabilidad configurada para restaurarla si se fuerza a 0 temporalmente
        self.configured_lava_prob = lava_prob
        self.key_prob = key_prob
        
        super().__init__(
            minNumRooms=n_rooms, 
            maxNumRooms=n_rooms, 
            maxRoomSize=8, 
            **kwargs
        )

    def _gen_grid(self, width, height):
        # 1. RESTAURAR CONFIGURACIÓN: Aseguramos que intentamos poner lava al inicio de cada episodio
        self.lava_prob = self.configured_lava_prob
        
        max_retries = 100 
        for _ in range(max_retries):
            
            self.grid = Grid(width, height)
            try:
                # Generación estándar de habitaciones
                super()._gen_grid(width, height)
            except Exception:
                continue

            # 2. Puertas y Llaves
            self._place_doors_probabilistic()

            # 3. Lava Inteligente
            self._add_lava_smart()

            # 4. Validación de Camino (BFS)
            if self._is_path_clear():
                return 

        # Si falla todo, generamos un nivel "fácil" sin lava para no crashear
        # Pero SOLO afecta a este episodio gracias a self.configured_lava_prob
        print(f"Warning: Could not generate a solvable level with lava after {max_retries} tries. Generating without lava.")
        self.lava_prob = 0.0
        super()._gen_grid(width, height)

    def _place_doors_probabilistic(self):
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
        
        for i, room in enumerate(self.rooms):
            if i == len(self.rooms) - 1: # Última habitación no tiene puerta de salida extra
                break
            
            # Decidimos si esta puerta se cierra con llave
            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                if door_pos:
                    color = random.choice(valid_colors)
                    # Puerta cerrada
                    self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                    # Llave en algún lugar de la habitación (max_tries alto para asegurar sitio)
                    self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)

    def _add_lava_smart(self):
        # Si la prob es 0, salimos rápido
        if self.lava_prob <= 0:
            return

        safe_cells = set()
        safe_cells.add(self.agent_pos)

        # A) Proteger Objetos (Llaves, Puertas, Meta) y su perímetro inmediato
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, (Key, Door, Goal)):
                    safe_cells.add((x, y))
                    # Marcar cruz adyacente como segura para facilitar la interacción
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((x + dx, y + dy))

        # B) Proteger Entradas/Salidas estructurales de las habitaciones
        for room in self.rooms:
            for d_pos in [room.entryDoorPos, room.exitDoorPos]:
                if d_pos:
                    safe_cells.add(d_pos)
                    # Asegurar paso libre alrededor de la puerta
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((d_pos[0]+dx, d_pos[1]+dy))

        # C) Colocar Lava
        for room in self.rooms:
            for x in range(room.top[0], room.top[0] + room.size[0]):
                for y in range(room.top[1], room.top[1] + room.size[1]):
                    if (x, y) not in safe_cells:
                        cell = self.grid.get(x, y)
                        if cell is None: # Solo en suelo vacío
                            if random.random() < self.lava_prob:
                                self.grid.set(x, y, Lava())

    def _is_path_clear(self):
        start = self.agent_pos
        end = None
        
        # Buscar la meta
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if isinstance(self.grid.get(x, y), Goal):
                    end = (x, y)
                    break
        if not end: return False 

        # BFS simple
        queue = collections.deque([start])
        visited = set([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == end: return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if (nx, ny) not in visited:
                        cell = self.grid.get(nx, ny)
                        # Podemos pasar si es None (suelo), Goal, Key, Door (asumimos que se pueden abrir)
                        # NO podemos pasar si es Lava o Wall
                        if not isinstance(cell, (Lava, Wall)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Living Penalty (pequeño incentivo para ir rápido)
        if reward == 0:
            reward = -0.005

        # Castigo fuerte por Lava (Terminated = Muerte)
        if terminated and reward <= 0:
            reward = -10.0  
            
        return obs, reward, terminated, truncated, info

# Registro del entorno
env_id = "MiniGrid-LavaSmartMulti-v0"
if env_id in gym.envs.registry:
    del gym.envs.registry[env_id]

register(
    id=env_id,
    entry_point=__name__ + ":CorredorLavaSmart",
)

# =============================================================================
# 2. IMAGE OBSERVATION WRAPPER
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # La observación "image" en MiniGrid es (7, 7, 3) codificada simbólicamente
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=img_space.shape,
            dtype="uint8"
        )

    def observation(self, obs):
        return obs["image"]

# =============================================================================
# 3. GRADUAL LAVA + MULTIKEY CURRICULUM TRAINING
# =============================================================================
def train_lava_smart_multi():
    
    initial_model_path = "Fase_4_Color_12Hab_FINAL4.zip" 
    
    # Configuración del Curriculum
    stages_config = [
        {"rooms": 1, "lava": 0.05, "key": 0.0, "steps": 500_000},
        {"rooms": 3, "lava": 0.05, "key": 0.10, "steps": 2_000_000},
        {"rooms": 6, "lava": 0.05, "key": 0.15, "steps": 2_000_000}, 
        {"rooms": 9, "lava": 0.05, "key": 0.20, "steps": 2_000_000},
        {"rooms": 12, "lava": 0.05, "key": 0.25, "steps": 2_000_000} 
    ]
    
    log_dir = "./tensorboard_logs2/"
    model = None 

    print("=== START TRAINING: LAVA SMART + MULTIPLE KEYS CURRICULUM ===")
    
    for i, config in enumerate(stages_config):
        n_rooms = config["rooms"]
        lava_prob = config["lava"]
        key_prob = config["key"]
        steps = config["steps"]
        
        stage_name = f"Lava2_Fase{i+1}_{n_rooms}Rooms"
        
        print(f"\n--- {stage_name} ---")
        print(f"  Rooms: {n_rooms} | Lava: {lava_prob*100}% | Keys: {key_prob*100}%")

        # Crear entorno para esta fase
        env = gym.make(
            env_id, 
            render_mode=None, 
            n_rooms=n_rooms, 
            lava_prob=lava_prob,
            key_prob=key_prob
        )
        env = ImgObsWrapper(env)

        # Lógica de carga / creación del modelo
        if model is None:
            # Fase 1: Intentar cargar o crear desde cero
            if os.path.exists(initial_model_path):
                print(f"  --> Loading pre-trained model: {initial_model_path}")
                custom_objects = {
                    "learning_rate": 0.0002, 
                    "ent_coef": 0.08,  
                    "clip_range": 0.2
                }
                model = PPO.load(
                    initial_model_path, 
                    env=env, 
                    custom_objects=custom_objects,
                    tensorboard_log=log_dir,
                    verbose=1,
                    device = 'cpu'
                )
            else:
                print("  --> No pre-trained model found. Creating new PPO (MlpPolicy).")
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    verbose=1, 
                    learning_rate=0.0003,
                    tensorboard_log=log_dir,
                    device="auto" # Usar GPU si está disponible
                )
        else:
            # Fases siguientes: Transferir el agente existente al nuevo entorno
            print(f"  --> Transferring agent to new environment (Rooms: {n_rooms})")
            model.set_env(env)

        # Configurar Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=f"./checkpoints2/{stage_name}/",
            name_prefix=stage_name
        )

        # ENTRENAMIENTO
        # IMPORTANTE: reset_num_timesteps=False permite que TensorBoard
        # muestre una gráfica continua sumando pasos, en vez de reiniciar a 0.
        model.learn(
            total_timesteps=steps, 
            callback=checkpoint_callback,
            reset_num_timesteps=True, 
            tb_log_name=stage_name
        )

        final_save_name = f"{stage_name}_FINAL"
        model.save(final_save_name)
        print(f"✅ Stage completed. Saved as {final_save_name}.zip")
        
        env.close()

    print("\nTRAINING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    train_lava_smart_multi()