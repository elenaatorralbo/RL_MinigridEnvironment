import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import os

# =============================================================================
# 1. DEFINICIÓN DEL ENTORNO MULTICOLOR
# =============================================================================
class CorredorMulticolor(MultiRoomEnv):
    def __init__(self, n_rooms=4, key_prob=0.2, **kwargs):
        super().__init__(
            minNumRooms=n_rooms, 
            maxNumRooms=n_rooms, 
            maxRoomSize=10, 
            **kwargs
        )
        self.key_prob = key_prob

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        # Lista de colores disponibles (evitamos verde 'green' porque son las puertas abiertas)
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']

        for i, room in enumerate(self.rooms):
            # La última habitación no tiene salida
            if i == len(self.rooms) - 1:
                break
            
            # 20% de probabilidad de puerta cerrada
            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                
                # --- NOVEDAD: ELEGIMOS COLOR ALEATORIO ---
                color = random.choice(valid_colors)
                
                # 1. Puerta cerrada de ese color
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                
                # 2. Llave del MISMO color en la habitación
                self.place_obj(
                    Key(color), 
                    top=room.top, 
                    size=room.size, 
                    max_tries=100
                )

# Registro del entorno
if "MiniGrid-CorredorMulticolor-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-CorredorMulticolor-v0"]

register(
    id="MiniGrid-CorredorMulticolor-v0",
    entry_point=__name__ + ":CorredorMulticolor",
)

# =============================================================================
# 2. WRAPPER DE IMAGEN
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
# 3. CURRICULUM GRADUAL MULTICOLOR CON TENSORBOARD
# =============================================================================
def run_multicolor_curriculum():
    
    # --- CONFIGURACIÓN ---
    # IMPORTANTE: Asegúrate de que este archivo existe o cámbialo por el último modelo bueno que tengas
    # Si quieres empezar desde el modelo de 6 habitaciones que ya funcionaba bien, pon su ruta aquí.
    # Si quieres empezar desde el experto en llaves original, pon "KeyDoor.zip".
    initial_model_path = "Fase_4_Color_12Hab_FINAL3.zip" 
    
    stages = [3, 6, 9, 12]   # Fases
    steps_per_stage = 2_000_000    # Pasos por fase
    log_dir = "./tensorboard_logs/" # Carpeta para los logs
    
    model = None 

    print("🚀 INICIANDO CURRICULUM MULTICOLOR: 3 -> 12 HABITACIONES 🚀")
    print(f"   Pasos por fase: {steps_per_stage}")
    print(f"   TensorBoard Logs: {log_dir}")
    print("   (Ejecuta 'tensorboard --logdir ./tensorboard_logs/' para ver las gráficas)")

    for i, n_rooms in enumerate(stages):
        stage_name = f"Fase_{i+1}_Color_{n_rooms}Hab"
        
        print(f"\n--------------------------------------------------")
        print(f"🏁 INICIANDO {stage_name}")
        print(f"   Objetivo: Cruzar {n_rooms} habitaciones (Colores variados)")
        print(f"--------------------------------------------------")

        # 1. Crear entorno
        env = gym.make("MiniGrid-CorredorMulticolor-v0", render_mode=None, n_rooms=n_rooms)
        env = ImgObsWrapper(env)

        # 2. Cargar / Transferir Modelo
        if model is None:
            if not os.path.exists(initial_model_path):
                print(f"❌ ERROR: No encuentro '{initial_model_path}'.")
                return
            
            print(f"🧠 Cargando cerebro base: {initial_model_path}")
            
            custom_objects = {
                "learning_rate": 0.0001,
                "ent_coef": 0.01
            }
            
            # AQUÍ AÑADIMOS TENSORBOARD
            model = PPO.load(
                initial_model_path, 
                env=env, 
                custom_objects=custom_objects,
                tensorboard_log=log_dir # <--- Activamos TensorBoard
            )
        else:
            print(f"🧠 Transfiriendo agente veterano a la nueva dificultad...")
            model.set_env(env)

        # 3. Checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./checkpoints/{stage_name}2/",
            name_prefix=stage_name
        )

        # 4. Entrenar
        # Usamos tb_log_name para separar las gráficas por fase dentro de TensorBoard
        model.learn(
            total_timesteps=steps_per_stage, 
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name=stage_name # <--- Nombre de la curva en la gráfica
        )

        # 5. Guardar final
        final_save_name = f"{stage_name}_FINAL3"
        model.save(final_save_name)
        print(f"✅ {stage_name} COMPLETADA. Guardado en {final_save_name}.zip")
        
        env.close()

    print("\n🏆 ¡MARATÓN MULTICOLOR COMPLETADO! Eres un maestro de las llaves.")

if __name__ == "__main__":
    run_multicolor_curriculum()