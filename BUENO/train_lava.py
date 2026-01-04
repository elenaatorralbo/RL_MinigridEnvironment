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

# =============================================================================
# 1. ENTORNO INTELIGENTE (LAVA SMART + MULTI KEYS)
# =============================================================================
class CorredorLavaSmart(MultiRoomEnv):
    def __init__(self, n_rooms=4, lava_prob=0.1, key_prob=0.1, **kwargs):
        super().__init__(
            minNumRooms=n_rooms, 
            maxNumRooms=n_rooms, 
            maxRoomSize=10, 
            **kwargs
        )
        self.key_prob = key_prob
        self.lava_prob = lava_prob

    def _gen_grid(self, width, height):
        # BUCLE DE SEGURIDAD: Reintentar si el nivel es imposible
        max_retries = 500
        for _ in range(max_retries):
            
            # 1. Base del Grid (Paredes y Habitaciones vacías)
            self.grid = Grid(width, height)
            try:
                super()._gen_grid(width, height)
            except Exception:
                continue

            # 2. AÑADIR PUERTAS Y LLAVES (Probabilístico en TODAS las habitaciones)
            self._place_doors_probabilistic()

            # 3. AÑADIR LAVA INTELIGENTE (Protege TODAS las llaves y puertas generadas)
            self._add_lava_smart()

            # 4. VALIDACIÓN DE CAMINO (BFS)
            if self._is_path_clear():
                return # ¡Nivel válido! Salimos del bucle.

        # Si falla todo, generamos uno sin lava para evitar crash
        print("⚠️ ALERTA: No se pudo generar nivel soluble. Generando sin lava.")
        self.lava_prob = 0.0
        super()._gen_grid(width, height)

    def _place_doors_probabilistic(self):
        """
        Itera por todas las habitaciones (menos la última) y decide
        si pone una puerta cerrada basándose en self.key_prob.
        """
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
        
        for i, room in enumerate(self.rooms):
            # La última habitación no tiene puerta de salida que bloquear
            if i == len(self.rooms) - 1:
                break
            
            # Decisión probabilística por cada habitación
            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)
                
                # Colocamos la Puerta Cerrada
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                
                # Colocamos la Llave en la misma habitación (para garantizar que es alcanzable antes de la puerta)
                self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)

    def _add_lava_smart(self):
        """
        Escanea el mapa buscando CUALQUIER Puerta o Llave.
        Marca sus posiciones y vecinos como 'SEGURAS'.
        Rellena el resto con lava al azar.
        """
        safe_cells = set()
        safe_cells.add(self.agent_pos)

        # A) Buscar Objetos Críticos (Llaves, Puertas, Meta)
        # Recorre todo el grid, así que detectará si hay 1 llave o 5 llaves.
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, (Key, Door, Goal)):
                    safe_cells.add((x, y))
                    # Añadimos vecinos para dar espacio de maniobra
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((x + dx, y + dy))

        # B) Asegurar puertas de entrada/salida de las habitaciones (huecos estructurales)
        for room in self.rooms:
            for d_pos in [room.entryDoorPos, room.exitDoorPos]:
                if d_pos:
                    safe_cells.add(d_pos)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((d_pos[0]+dx, d_pos[1]+dy))

        # C) Poner Lava donde NO sea seguro
        for room in self.rooms:
            # Iterar solo dentro de la habitación
            for x in range(room.top[0], room.top[0] + room.size[0]):
                for y in range(room.top[1], room.top[1] + room.size[1]):
                    
                    cell = self.grid.get(x, y)
                    # Si está vacío Y no es zona segura
                    if cell is None and (x, y) not in safe_cells:
                        if random.random() < self.lava_prob:
                            self.grid.set(x, y, Lava())

    def _is_path_clear(self):
        """BFS simple: Comprueba conectividad física Start -> Goal"""
        start = self.agent_pos
        end = None
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if isinstance(self.grid.get(x, y), Goal):
                    end = (x, y)
                    break
        if not end: return False 

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
                        # Caminable: Vacío, Meta, Llave, Puerta. NO LAVA.
                        if not isinstance(cell, (Lava, Wall)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False

# Registro
if "MiniGrid-LavaSmartMulti-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-LavaSmartMulti-v0"]

register(
    id="MiniGrid-LavaSmartMulti-v0",
    entry_point=__name__ + ":CorredorLavaSmart",
)

# =============================================================================
# 2. WRAPPER
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=img_space.shape, dtype="uint8"
        )
    def observation(self, obs):
        return obs["image"]

# =============================================================================
# 3. CURRICULUM DE ENTRENAMIENTO
# =============================================================================
def train_lava_smart_multi():
    
    # ⚠️ IMPORTANTE: Ruta del modelo anterior
    initial_model_path = "Fase_4_Color_12Hab_FINAL4.zip" 
    
    # Configuración del Curriculum:
    # Aumentamos Rooms, Probabilidad de Lava Y Probabilidad de Llaves simultáneamente.
    stages_config = [
        {"rooms": 3,  "lava": 0.10, "key": 0.10, "steps": 2_000_000},
        {"rooms": 6,  "lava": 0.15, "key": 0.15, "steps": 2_000_000},
        {"rooms": 9,  "lava": 0.20, "key": 0.20, "steps": 2_000_000},
        {"rooms": 12, "lava": 0.25, "key": 0.25, "steps": 2_000_000} 
    ]
    
    log_dir = "./tensorboard_logs/"
    model = None 

    print("🚀 INICIANDO ENTRENAMIENTO: LAVA SMART + MULTIPLE KEYS 🚀")
    
    for i, config in enumerate(stages_config):
        n_rooms = config["rooms"]
        lava_prob = config["lava"]
        key_prob = config["key"]
        steps = config["steps"]
        
        # Nombre descriptivo para TensorBoard y guardado
        stage_name = f"MultiKey_Fase{i+1}_R{n_rooms}_L{int(lava_prob*100)}_K{int(key_prob*100)}"
        
        print(f"\n--------------------------------------------------")
        print(f"🔥 {stage_name}")
        print(f"   Habitaciones: {n_rooms}")
        print(f"   Probabilidad Lava: {lava_prob*100}%")
        print(f"   Probabilidad Llaves: {key_prob*100}% (Puede haber múltiples)")
        print(f"--------------------------------------------------")

        # Crear Entorno pasando ambas probabilidades
        env = gym.make(
            "MiniGrid-LavaSmartMulti-v0", 
            render_mode=None, 
            n_rooms=n_rooms, 
            lava_prob=lava_prob,
            key_prob=key_prob
        )
        env = ImgObsWrapper(env)

        # Cargar Modelo
        if model is None:
            if not os.path.exists(initial_model_path):
                print(f"❌ ERROR: No encuentro '{initial_model_path}'.")
                return
            
            print(f"🧠 Cargando modelo previo: {initial_model_path}")
            custom_objects = {"learning_rate": 0.0001, "ent_coef": 0.01}
            
            model = PPO.load(
                initial_model_path, 
                env=env, 
                custom_objects=custom_objects,
                tensorboard_log=log_dir
            )
        else:
            print(f"🧠 Transfiriendo agente a la siguiente fase...")
            model.set_env(env)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./checkpoints_multikey/{stage_name}/",
            name_prefix=stage_name
        )

        # Entrenar
        model.learn(
            total_timesteps=steps, 
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name=stage_name
        )

        final_save_name = f"{stage_name}_FINAL"
        model.save(final_save_name)
        print(f"✅ Fase completada. Guardado en {final_save_name}.zip")
        env.close()

    print("\n🏆 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO! 🏆")

if __name__ == "__main__":
    train_lava_smart_multi()