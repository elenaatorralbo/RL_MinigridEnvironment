import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces, ObservationWrapper
import os

# =============================================================================
# 1. WRAPPER (EL MISMO)
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
# 2. ENTRENAMIENTO DE HABILIDAD "USAR LLAVES" + TENSORBOARD
# =============================================================================
def train_doorkey():
    
    # Entorno oficial de Minigrid para aprender a usar llaves
    env_id = "MiniGrid-DoorKey-8x8-v0"
    
    # Ruta de tu modelo experto en navegación
    prev_model_path = os.path.join("checkpoints", "Nivel_2_5_Intermedio", "Nivel_2_5_Intermedio_500000_steps.zip")
    new_model_name = "KeyDoor"

    # ### NUEVO: Definimos la carpeta para los logs
    log_dir = "./tensorboard_logs_keys/"

    print(f"--- 🔑 INICIANDO ENTRENAMIENTO DE LLAVES 🔑 ---")
    print(f"Objetivo: Aprender acciones 'Pickup' y 'Toggle'")
    print(f"Logs de TensorBoard en: {log_dir}")

    # 1. Crear entorno DoorKey
    env = gym.make(env_id, render_mode=None)
    env = ImgObsWrapper(env)

    # 2. Cargar cerebro previo
    if os.path.exists(prev_model_path):
        print(f"🧠 Cargando cerebro experto en navegación: {prev_model_path}")
        
        custom_objects = {
            "learning_rate": 0.0001,
            "ent_coef": 0.1 
        }
        
        # ### NUEVO: Añadimos tensorboard_log al cargar
        model = PPO.load(
            prev_model_path, 
            env=env, 
            custom_objects=custom_objects,
            tensorboard_log=log_dir # <--- Aquí activamos el logger
        )
    else:
        print("⚠️ No encuentro el modelo anterior. Empezando de cero.")
        # ### NUEVO: También aquí por si acaso
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # 3. Guardado
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"./checkpoints/{new_model_name}/",
        name_prefix=new_model_name
    )

    # 4. Entrenar
    model.learn(
        total_timesteps=500_000, 
        callback=checkpoint_callback,
        reset_num_timesteps=True,
        tb_log_name="Entrenamiento_Llaves" # ### NUEVO: Nombre de la curva en la gráfica
    )
    
    model.save(new_model_name)
    print("--- 🔓 ENTRENAMIENTO DE LLAVES COMPLETADO ---")

if __name__ == "__main__":
    train_doorkey()