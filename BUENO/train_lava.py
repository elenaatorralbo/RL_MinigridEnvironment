import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import os
import collections

# =============================================================================
# 1. ENTORNO CON LAVA + MAX 1 PUERTA + PATHFINDING
# =============================================================================
class CorredorConLava(MultiRoomEnv):
    def __init__(self, n_rooms=4, lava_prob=0.1, key_prob=0.2, **kwargs):
        super().__init__(
            minNumRooms=n_rooms, 
            maxNumRooms=n_rooms, 
            maxRoomSize=10, 
            **kwargs
        )
        self.key_prob = key_prob
        self.lava_prob = lava_prob

    def _gen_grid(self, width, height):
        # Bucle de seguridad: Seguimos intentando hasta que salga un nivel posible
        while True:
            try:
                super()._gen_grid(width, height)
            except Exception:
                continue

            # 2. AÑADIMOS PUERTAS (LÓGICA "MÁXIMO UNA")
            valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
            locked_door_placed = False # <--- VARIABLE DE CONTROL
            
            # Barajamos las habitaciones para que la puerta no siempre salga al principio
            # (Hacemos una lista de índices y la desordenamos)
            room_indices = list(range(len(self.rooms) - 1)) # -1 porque la última no tiene salida
            random.shuffle(room_indices)

            for i in room_indices:
                room = self.rooms[i]
                
                # Si AÚN NO hemos puesto puerta Y el dado del 20% sale...
                if not locked_door_placed and random.random() < self.key_prob:
                    door_pos = room.exitDoorPos
                    color = random.choice(valid_colors)
                    
                    # Ponemos la puerta cerrada
                    self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                    
                    # Ponemos la llave
                    self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)
                    
                    # ¡IMPORTANTE! Marcamos que ya hay una.
                    locked_door_placed = True 

            # 3. AÑADIMOS LAVA (Esto sí aumenta con las fases)
            for x in range(0, width):
                for y in range(0, height):
                    cell = self.grid.get(x, y)
                    # Solo ponemos lava en suelo vacío
                    if cell is None and (x, y) != self.agent_pos:
                        if random.random() < self.lava_prob:
                            self.grid.set(x, y, Lava())

            # 4. VALIDACIÓN DE CAMINO (BFS)
            if self.is_path_clear():
                break # Nivel válido, salimos
            else:
                self.grid = None # Nivel imposible, regeneramos

    def is_path_clear(self):
        """BFS para asegurar que la lava no bloquea el paso totalmente."""
        start = self.agent_pos
        goal_pos = None
        for x in range(self.width):
            for y in range(self.height):
                if isinstance(self.grid.get(x, y), Goal):
                    goal_pos = (x, y)
                    break
            if goal_pos: break
        
        if not goal_pos: return False

        queue = collections.deque([start])
        visited = set()
        visited.add(start)

        while queue:
            curr_x, curr_y = queue.popleft()
            if (curr_x, curr_y) == goal_pos:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_x + dx, curr_y + dy
                
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited:
                        obj = self.grid.get(nx, ny)
                        # La lava NO es caminable
                        if not isinstance(obj, (Wall, Lava)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False

# Registro
if "MiniGrid-CorredorLavaMax1-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-CorredorLavaMax1-v0"]

register(
    id="MiniGrid-CorredorLavaMax1-v0",
    entry_point=__name__ + ":CorredorConLava",
)

# =============================================================================
# 2. WRAPPER
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
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
# 3. CURRICULUM LAVA DEFINITIVO
# =============================================================================
def run_lava_curriculum_max1():
    
    # RUTA: El modelo final de la Fase Multicolor
    initial_model_path = "Fase_4_Color_12Hab_FINAL2.zip" 
    
    # CONFIGURACIÓN PROGRESIVA (Habitaciones + Probabilidad Lava)
    stages_config = [
        {"rooms": 3,  "lava": 0.10, "steps": 500_000},
        {"rooms": 6,  "lava": 0.15, "steps": 500_000},
        {"rooms": 9,  "lava": 0.20, "steps": 500_000},
        {"rooms": 12, "lava": 0.25, "steps": 1_000_000} # La última fase es más larga (1M)
    ]
    
    log_dir = "./tensorboard_logs/"
    model = None 

    print("🚀 INICIANDO CURRICULUM LAVA (MAX 1 PUERTA) 🚀")
    
    for i, config in enumerate(stages_config):
        n_rooms = config["rooms"]
        lava_prob = config["lava"]
        steps = config["steps"]
        
        stage_name = f"Lava_Max1_Fase{i+1}_R{n_rooms}_L{int(lava_prob*100)}"
        
        print(f"\n--------------------------------------------------")
        print(f"🌋 {stage_name}")
        print(f"   Habitaciones: {n_rooms} | Lava: {lava_prob*100}% | Pasos: {steps}")
        print(f"--------------------------------------------------")

        env = gym.make(
            "MiniGrid-CorredorLavaMax1-v0", 
            render_mode=None, 
            n_rooms=n_rooms, 
            lava_prob=lava_prob
        )
        env = ImgObsWrapper(env)

        # Cargar / Transferir
        if model is None:
            if not os.path.exists(initial_model_path):
                print(f"❌ ERROR: No encuentro '{initial_model_path}'.")
                return
            
            print(f"🧠 Cargando modelo base: {initial_model_path}")
            
            custom_objects = {
                "learning_rate": 0.0001,
                "ent_coef": 0.01
            }
            
            model = PPO.load(
                initial_model_path, 
                env=env, 
                custom_objects=custom_objects,
                tensorboard_log=log_dir
            )
        else:
            print(f"🧠 Transfiriendo agente...")
            model.set_env(env)

        # Checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./checkpoints_lava/{stage_name}/",
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
        print(f"✅ Completado. Guardado en {final_save_name}.zip")
        env.close()

    print("\n🏆 ¡SUPERVIVIENTE DEFINITIVO! Agente experto en Lava.")

if __name__ == "__main__":
    run_lava_curriculum_max1()